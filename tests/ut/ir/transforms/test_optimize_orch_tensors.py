# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for OptimizeOrchTensors pass.

Each test uses explicit Before (post-ConvertTensorToTileOps tile-level IR)
and Expected (optimized) programs in @pl.program style.
"""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _call_name(expr) -> str:
    if not isinstance(expr, ir.Call):
        return ""
    return getattr(getattr(expr, "op", None), "name", "")


def _is_call_named(expr, *names: str) -> bool:
    return _call_name(expr) in names


class TestIterArgReuse:
    """Pattern 1: Merge Out params into In params via iter-arg feedback."""

    def test_simple_single_return(self):
        """Single-return InCore in ForStmt: Out param merged into InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, ret0__out)
                    new_acc = pl.yield_(result)
                return new_acc

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.InOut[pl.Tensor[[64], pl.FP32]],
                x: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile = pl.load(acc, [0], [64])
                x__tile = pl.load(x, [0], [64])
                y__tile = pl.tile.add(acc__tile, x__tile)
                ret0__store = pl.store(y__tile, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    result = self.main_incore_0(acc, x)
                    new_acc = pl.yield_(result)
                return new_acc

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_iter_arg_merge_preserves_dump_vars(self):
        """The Out->InOut merge rewrites the incore call site; a ``kAttrDumpVars``
        tag on a surviving (non-merged) In arg must ride through the rewrite.

        Regression: ``CallSiteRewriter::VisitStmt_`` rebuilt the call with a Call
        constructor that drops ``attrs_``, so ``pl.dump_tag``-seeded ``dump_vars``
        was lost. ``x`` is loop-invariant (same Var across iterations) and is
        consumed by the in-loop dispatch but is NOT the merged Out param, so its
        dump tag must survive the merge."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(acc__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self, acc0: pl.Tensor[[64], pl.FP32], x: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                pl.dump_tag(x)
                for i, (acc,) in pl.range(10, init_values=(acc0,)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, ret0__out)
                    new_acc = pl.yield_(result)
                return new_acc

        After = passes.optimize_orch_tensors()(Before)

        dump_var_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                name = _call_name(op)
                if name == "main_incore_0":
                    dv = (op.attrs or {}).get("dump_vars")
                    if dv:
                        dump_var_names.extend(v.name_hint.split("__", 1)[0] for v in dv)
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "x" in dump_var_names, (
            f"dump_vars dropped by the iter-arg-merge call rewrite; got {dump_var_names}"
        )

    def test_multi_return_iter_arg(self):
        """Multi-return InCore with two iter-arg-fed Out params: both merged to InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
                ret1__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                z__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(z__tile, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, ret0__out, ret1__out
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.InOut[pl.Tensor[[64], pl.FP32]],
                b: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile = pl.load(a, [0], [64])
                b__tile = pl.load(b, [0], [64])
                y__tile = pl.tile.add(a__tile, b__tile)
                z__tile = pl.tile.mul(a__tile, b__tile)
                ret0__store = pl.store(y__tile, [0], a)
                ret1__store = pl.store(z__tile, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result = self.main_incore_0(a, b)
                    new_a = result[0]
                    new_b = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multi_return_with_if_branch(self):
        """Multi-return InCore with IfStmt branch: Out params merged to InOut."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[64], pl.FP32],
                b: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
                ret1__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile: pl.Tile[[64], pl.FP32] = pl.load(a, [0], [64])
                b__tile: pl.Tile[[64], pl.FP32] = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_a, [0], ret0__out)
                ret1__store: pl.Tensor[[64], pl.FP32] = pl.store(phi_b, [0], ret1__out)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    result: tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]] = self.main_incore_0(
                        a, b, n, ret0__out, ret1__out
                    )
                    new_a: pl.Tensor[[64], pl.FP32] = result[0]
                    new_b: pl.Tensor[[64], pl.FP32] = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.InOut[pl.Tensor[[64], pl.FP32]],
                b: pl.InOut[pl.Tensor[[64], pl.FP32]],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                a__tile = pl.load(a, [0], [64])
                b__tile = pl.load(b, [0], [64])
                if n == 0:
                    ra: pl.Tile[[64], pl.FP32] = a__tile
                    rb: pl.Tile[[64], pl.FP32] = b__tile
                    phi_a, phi_b = pl.yield_(ra, rb)
                else:
                    ra__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a__tile, b__tile)
                    rb__tile: pl.Tile[[64], pl.FP32] = pl.tile.mul(a__tile, b__tile)
                    phi_a, phi_b = pl.yield_(ra__tile, rb__tile)
                ret0__store = pl.store(phi_a, [0], a)
                ret1__store = pl.store(phi_b, [0], b)
                return ret0__store, ret1__store

            @pl.function
            def main(
                self,
                a0: pl.Tensor[[64], pl.FP32],
                b0: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
                for i, (a, b) in pl.range(3, init_values=(a0, b0)):
                    result = self.main_incore_0(a, b, n)
                    new_a = result[0]
                    new_b = result[1]
                    out_a, out_b = pl.yield_(new_a, new_b)
                return out_a, out_b

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_standalone_call_merges_in_out(self):
        """Standalone InCore call with an iter_arg chain (remainder-kernel shape):
        In + tensor.create Out pair merges to InOut even without an enclosing loop.

        Regression for #928: pl.parallel remainder kernel lost inout accumulation
        because Pattern 1 only matched calls inside an iter-arg loop.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    new_a__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(a, x__tile)
                    final = pl.yield_(new_a__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, x, n, ret0__out)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.InOut[pl.Tensor[[64], pl.FP32]],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile = pl.load(acc, [0], [64])
                x__tile = pl.load(x, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    new_a__tile = pl.tile.add(a, x__tile)
                    final = pl.yield_(new_a__tile)
                ret0__store = pl.store(final, [0], acc)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                result = self.main_incore_0(acc, x, n)
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_standalone_call_in_arg_reused_not_merged(self):
        """Safety: when the In arg is read again after the call, do NOT merge.

        Merging would clobber the original value the later use expects.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    next_a: pl.Tile[[64], pl.FP32] = pl.tile.add(a, a)
                    final = pl.yield_(next_a)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(acc__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                _unused: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret0__out)
                ret1__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.reader(acc, ret1__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # acc is read again by reader — merging main_incore_0's In/Out would
        # corrupt it. Expected: Before is unchanged.
        ir.assert_structural_equal(After, Before)

    def test_standalone_call_unsafe_sibling_blocks_merge(self):
        """When the same callee has multiple standalone call sites, the merge
        must only apply if EVERY site is safe. One unsafe sibling (here: the
        second call reuses `acc` after a later call) must block the rewrite —
        otherwise the rewrite corrupts the sibling's In arg.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                acc__tile: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                for i, (a,) in pl.range(n, init_values=(acc__tile,)):
                    next_a: pl.Tile[[64], pl.FP32] = pl.tile.add(a, a)
                    final = pl.yield_(next_a)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(final, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self,
                acc: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INT64],
            ) -> pl.Tensor[[64], pl.FP32]:
                # First call: acc is read again below → unsafe to merge.
                ret_a: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                _first: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret_a)
                # Second call: uses acc again (this is the "unsafe" sibling).
                ret_b: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.main_incore_0(acc, n, ret_b)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # Any rewrite here would silently corrupt at least one of the two
        # callers, so the pass must leave Before untouched.
        ir.assert_structural_equal(After, Before)

    def test_standalone_call_without_iter_arg_chain_not_merged(self):
        """A standalone call whose callee is a plain load→store (no iter_arg
        chain) is NOT merged: we require semantic evidence (an iter_arg chain)
        that the In/Out were intended to alias.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_copy(
                self,
                src: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(src, [0], [64])
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, src: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                result: pl.Tensor[[64], pl.FP32] = self.kernel_copy(src, ret0__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        # kernel_copy has no iter_arg loop → no merge expected.
        ir.assert_structural_equal(After, Before)

    def test_no_iter_arg_no_change(self):
        """InCore call not in iter-arg loop: no optimization, Out params remain."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # No iter-arg loop → should be unchanged
        ir.assert_structural_equal(After, Before)


class TestLoopHoisting:
    """Loop hoisting (disabled — breaks scope-based alloc_tensors batching)."""

    def test_tensor_create_stays_inside_loop(self):
        """tensor.create stays inside loop to preserve scope-based memory batching."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                x__tile: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                y__tile: pl.Tile[[64], pl.FP32] = pl.tile.add(x__tile, x__tile)
                ret0__store: pl.Tensor[[64], pl.FP32] = pl.store(y__tile, [0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(10):
                    ret0__out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                    y: pl.Tensor[[64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # Loop hoisting disabled: tensor.create should remain unchanged
        ir.assert_structural_equal(After, Before)


class TestAssembleParentStrides:
    """Pattern 2: Attach parent-derived strides to Out params for assemble patterns."""

    def test_out_param_gets_parent_stride(self):
        """When InCore result feeds tensor.assemble in orch, Out param gets parent strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [mb, nb], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(a, mb, nb, ret0__out)
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(c_iter2, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile = pl.load(a, [mb, nb], [32, 32])
                ret0__store: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                    for nb, (c_iter2,) in pl.range(0, 128, 32, init_values=(c_iter,)):
                        ret0__out = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result = self.main_incore_0(
                            a, mb, nb, ret0__out
                        )
                        c_next = pl.assemble(
                            c_iter2, result, [mb, nb]
                        )
                        c_rv = pl.yield_(c_next)
                    c_rv2 = pl.yield_(c_rv)
                return c_rv2
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_3d_parent_out_param_gets_trailing_stride(self):
        """When parent tensor is 3D and output tile is 2D, only trailing strides are applied."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q0: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, q0], [16, 64])
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q_proj: pl.Out[pl.Tensor[[4, 128, 5120], pl.FP32]],
            ) -> pl.Tensor[[4, 128, 5120], pl.FP32]:
                for b in pl.range(4):
                    for p0 in pl.range(0, 128, 16):
                        for q0, (q_iter,) in pl.range(0, 5120, 64, init_values=(q_proj,)):
                            ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor(
                                [16, 64], dtype=pl.FP32
                            )
                            result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(x, q0, ret0__out)
                            q_next: pl.Tensor[[4, 128, 5120], pl.FP32] = pl.assemble(
                                q_iter, result, [b, p0, q0]
                            )
                            q_rv = pl.yield_(q_next)
                return q_rv

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q0: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile = pl.load(x, [0, q0], [16, 64])
                ret0__store: pl.Tensor[  # noqa: E501
                    [16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                x: pl.Tensor[[16, 5120], pl.FP32],
                q_proj: pl.Out[pl.Tensor[[4, 128, 5120], pl.FP32]],
            ) -> pl.Tensor[[4, 128, 5120], pl.FP32]:
                for b in pl.range(4):
                    for p0 in pl.range(0, 128, 16):
                        for q0, (q_iter,) in pl.range(0, 5120, 64, init_values=(q_proj,)):
                            ret0__out = pl.create_tensor(
                                [16, 64], dtype=pl.FP32
                            )
                            result = self.proj_incore_0(
                                x, q0, ret0__out
                            )
                            q_next = pl.assemble(
                                q_iter, result, [b, p0, q0]
                            )
                            q_rv = pl.yield_(q_next)
                return q_rv
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)


class TestAssembleLoopRewrite:
    """Pattern 3: Rewrite tile.assemble loops to tile.store loops."""

    def test_assemble_loop_to_store_loop(self):
        """ForStmt with tile.assemble rewritten to tile.store with Out param as iter-arg init."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                buf__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.create(
                    [1, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                for i, (acc,) in pl.range(2, init_values=(buf__tile,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(x, [0, 0], [1, 32])
                    acc_next__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.assemble(acc, chunk__tile, [0, off])
                    result: pl.Tile[[1, 64], pl.FP32] = pl.yield_(acc_next__tile)
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(result, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, ret0__out)
                return y

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                for i, (acc,) in pl.range(2, init_values=(ret0__out,)):
                    off: pl.Scalar[pl.INDEX] = i * 32
                    chunk__tile = pl.load(x, [0, 0], [1, 32])
                    acc_next = pl.store(chunk__tile, [0, off], acc)
                    result = pl.yield_(acc_next)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[1, 32], pl.FP32]) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out = pl.create_tensor([1, 64], dtype=pl.FP32)
                y = self.main_incore_0(x, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)


class TestSliceInputStrides:
    """Pattern 4: Attach parent-derived strides to In params for slice patterns."""

    def test_in_param_gets_parent_stride_from_slice(self):
        """When orch slices a 2D parent and passes result to InCore In param, param gets parent strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(a, [0, 0], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb in pl.range(0, 128, 32):
                    for nb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                        chunk: pl.Tensor[[32, 32], pl.FP32] = pl.slice(data, [32, 32], [mb, nb])
                        ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                        result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(chunk, mb, nb, ret0__out)
                        c_next: pl.Tensor[[128, 128], pl.FP32] = pl.assemble(c_iter, result, [mb, nb])
                        c_rv = pl.yield_(c_next)
                return c_rv

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                a: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ],
                mb: pl.Scalar[pl.INDEX],
                nb: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[  # noqa: E501
                    pl.Tensor[[32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)]
                ],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                a__tile = pl.load(a, [0, 0], [32, 32])
                ret0__store: pl.Tensor[  # noqa: E501
                    [32, 32], pl.FP32, pl.TensorView(stride=[128, 1], layout=pl.TensorLayout.ND)
                ] = pl.store(a__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                for mb in pl.range(0, 128, 32):
                    for nb, (c_iter,) in pl.range(0, 128, 32, init_values=(c,)):
                        chunk = pl.slice(data, [32, 32], [mb, nb])
                        ret0__out = pl.create_tensor(
                            [32, 32], dtype=pl.FP32
                        )
                        result = self.main_incore_0(
                            chunk, mb, nb, ret0__out
                        )
                        c_next = pl.assemble(
                            c_iter, result, [mb, nb]
                        )
                        c_rv = pl.yield_(c_next)
                return c_rv
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_3d_parent_in_param_gets_trailing_stride(self):
        """When parent tensor is 3D and input slice is 2D, only trailing strides are applied."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[[16, 64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile: pl.Tile[[16, 64], pl.FP32] = pl.load(x, [0, 0], [16, 64])
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                data: pl.Tensor[[4, 128, 5120], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                chunk: pl.Tensor[[16, 64], pl.FP32] = pl.slice(data, [16, 64], [0, 0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                result: pl.Tensor[[16, 64], pl.FP32] = self.proj_incore_0(chunk, ret0__out)
                return result

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def proj_incore_0(
                self,
                x: pl.Tensor[  # noqa: E501
                    [16, 64], pl.FP32, pl.TensorView(stride=[5120, 1], layout=pl.TensorLayout.ND)
                ],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                x__tile = pl.load(x, [0, 0], [16, 64])
                ret0__store = pl.store(
                    x__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def proj(
                self,
                data: pl.Tensor[[4, 128, 5120], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                chunk = pl.slice(data, [16, 64], [0, 0, 0])
                ret0__out = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result = self.proj_incore_0(chunk, ret0__out)
                return result
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_multiple_sliced_in_params(self):
        """Multiple In params from different parents each get correct strides."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm_incore_0(
                self,
                a: pl.Tensor[[16, 128], pl.FP32],
                b: pl.Tensor[[128, 64], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a__tile: pl.Tile[[16, 128], pl.FP32] = pl.load(a, [0, 0], [16, 128])
                b__tile: pl.Tile[[128, 64], pl.FP32] = pl.load(b, [0, 0], [128, 64])
                c__tile: pl.Tile[[16, 64], pl.FP32] = pl.tile.matmul(a__tile, b__tile)
                ret0__store: pl.Tensor[[16, 64], pl.FP32] = pl.store(c__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def gemm(
                self,
                attn_out: pl.Tensor[[16, 8192], pl.FP32],
                wo: pl.Tensor[[8192, 8192], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a_chunk: pl.Tensor[[16, 128], pl.FP32] = pl.slice(attn_out, [16, 128], [0, 0])
                w_chunk: pl.Tensor[[128, 64], pl.FP32] = pl.slice(wo, [128, 64], [0, 0])
                ret0__out: pl.Tensor[[16, 64], pl.FP32] = pl.create_tensor([16, 64], dtype=pl.FP32)
                result: pl.Tensor[[16, 64], pl.FP32] = self.gemm_incore_0(a_chunk, w_chunk, ret0__out)
                return result

        # fmt: off
        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def gemm_incore_0(
                self,
                a: pl.Tensor[  # noqa: E501
                    [16, 128], pl.FP32, pl.TensorView(stride=[8192, 1], layout=pl.TensorLayout.ND)
                ],
                b: pl.Tensor[  # noqa: E501
                    [128, 64], pl.FP32, pl.TensorView(stride=[8192, 1], layout=pl.TensorLayout.ND)
                ],
                ret0__out: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a__tile = pl.load(a, [0, 0], [16, 128])
                b__tile = pl.load(b, [0, 0], [128, 64])
                c__tile = pl.tile.matmul(a__tile, b__tile)
                ret0__store = pl.store(
                    c__tile, [0, 0], ret0__out
                )
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def gemm(
                self,
                attn_out: pl.Tensor[[16, 8192], pl.FP32],
                wo: pl.Tensor[[8192, 8192], pl.FP32],
            ) -> pl.Tensor[[16, 64], pl.FP32]:
                a_chunk = pl.slice(
                    attn_out, [16, 128], [0, 0]
                )
                w_chunk = pl.slice(wo, [128, 64], [0, 0])
                ret0__out = pl.create_tensor(
                    [16, 64], dtype=pl.FP32
                )
                result = self.gemm_incore_0(
                    a_chunk, w_chunk, ret0__out
                )
                return result
        # fmt: on

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_non_sliced_in_param_unchanged(self):
        """In params that are not from tensor.slice remain unchanged."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[32, 32], pl.FP32],
                ret0__out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                x__tile: pl.Tile[[32, 32], pl.FP32] = pl.load(x, [0, 0], [32, 32])
                ret0__store: pl.Tensor[[32, 32], pl.FP32] = pl.store(x__tile, [0, 0], ret0__out)
                return ret0__store

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                ret0__out: pl.Tensor[[32, 32], pl.FP32] = pl.create_tensor([32, 32], dtype=pl.FP32)
                result: pl.Tensor[[32, 32], pl.FP32] = self.main_incore_0(data, ret0__out)
                return result

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)


class TestEdgeCases:
    """Edge cases: pass should not modify programs that don't match any pattern."""

    def test_no_incore_functions(self):
        """Programs with no InCore functions pass through unchanged."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y: pl.Tensor[[64], pl.FP32] = pl.add(x, x)
                return y

        After = passes.optimize_orch_tensors()(Before)
        ir.assert_structural_equal(After, Before)


class TestStaticOutputWindows:
    """Pattern 5: static output windows are precomputed at the parent tensor."""

    @pytest.fixture(autouse=True)
    def _no_roundtrip_verification(self):
        """The static-window planner emits pass-internal tensor ops that are
        consumed by orchestration codegen, not by the Python DSL parser.
        Keep property verification, but skip print -> parse roundtrip here.
        """
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_direct_out_call_uses_precomputed_window_ops(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_stripe(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                out: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                row: pl.Scalar[pl.INDEX] = 64
                out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel_stripe(data, row, 1.0, out)
                return out_next

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("kernel_stripe__windowed") is not None

        call_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "tensor.precompute_static_windows" in call_names
        assert "tensor.static_window_get" in call_names
        assert "tensor.slice" not in call_names


class TestPattern3WhileLoop:
    """Pattern 3 (AssembleLoopRewriter) is ForStmt-only.

    The rewriter (LoopRewriteMutator) only overrides VisitStmt_(ForStmtPtr)
    (src ~line 1328); there is no WhileStmt branch. So a while-carried
    tile.assemble accumulation must stay baseline: the tile.create buffer is
    kept, the iter-arg init stays the buffer (not the Out param), and the
    tile.assemble is NOT rewritten to tile.store. This is the dual of the
    passing ForStmt case in TestAssembleLoopRewrite.test_assemble_loop_to_store_loop.
    """

    def test_while_assemble_loop_not_rewritten(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main_incore_0(
                self,
                x: pl.Tensor[[1, 32], pl.FP32],
                n: pl.Scalar[pl.INDEX],
                ret0__out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                buf__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.create(
                    [1, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
                )
                i0: pl.Scalar[pl.INDEX] = 0
                for acc, ii in pl.while_(init_values=(buf__tile, i0)):
                    pl.cond(ii < n)
                    off: pl.Scalar[pl.INDEX] = ii * 32
                    chunk__tile: pl.Tile[[1, 32], pl.FP32] = pl.load(x, [0, 0], [1, 32])
                    acc_next__tile: pl.Tile[[1, 64], pl.FP32] = pl.tile.assemble(acc, chunk__tile, [0, off])
                    ii_next: pl.Scalar[pl.INDEX] = ii + 1
                    acc_rv, ii_rv = pl.yield_(acc_next__tile, ii_next)
                ret0__store: pl.Tensor[[1, 64], pl.FP32] = pl.store(acc_rv, [0, 0], ret0__out)
                return ret0__store

            @pl.function
            def main(
                self, x: pl.Tensor[[1, 32], pl.FP32], n: pl.Scalar[pl.INDEX]
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                ret0__out: pl.Tensor[[1, 64], pl.FP32] = pl.create_tensor([1, 64], dtype=pl.FP32)
                y: pl.Tensor[[1, 64], pl.FP32] = self.main_incore_0(x, n, ret0__out)
                return y

        After = passes.optimize_orch_tensors()(Before)
        # Pattern 3 only matches ForStmt; the WhileStmt assemble loop is left
        # untouched. (Patterns 1/4 also do not fire: the In param x is sliced
        # nowhere, and there is no iter-arg-fed In/Out merge.)
        ir.assert_structural_equal(After, Before)


class TestWindowDependencyPlanner:
    """WindowDependencyPlanner: barrier insertion and Array[TASK_ID] carry."""

    @pytest.fixture(autouse=True)
    def _no_roundtrip_verification(self):
        from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_alias_tracker_reshapes_to_same_root(self):
        """tensor.reshape and tensor.assemble chain resolves to the same root."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                out: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                row: pl.Scalar[pl.INDEX] = 64
                out_next: pl.Tensor[[256, 64], pl.FP32] = self.kernel(data, row, 1.0, out)
                return out_next

        After = passes.optimize_orch_tensors()(Before)
        windowed = After.get_function("kernel__windowed")
        assert windowed is not None

        call_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "tensor.precompute_static_windows" in call_names
        assert "tensor.static_window_get" in call_names

    def test_different_roots_no_false_dep(self):
        """Two windowed writes to different parents must not produce cross-deps.

        If q_proj writes to data_q and k_proj writes to data_k,
        no task_dummy barrier should appear (different storage roots).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel_q(
                self,
                data: pl.Tensor[[128, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[128, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def kernel_k(
                self,
                data: pl.Tensor[[128, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                bias: pl.Scalar[pl.FP32],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, bias)
                ret: pl.Tensor[[128, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data_q: pl.Tensor[[128, 64], pl.FP32],
                data_k: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                out_q: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                out_k: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                row: pl.Scalar[pl.INDEX] = 0
                result_q: pl.Tensor[[128, 64], pl.FP32] = self.kernel_q(data_q, row, 1.0, out_q)
                result_k: pl.Tensor[[128, 64], pl.FP32] = self.kernel_k(data_k, row, 1.0, out_k)
                return result_k

        After = passes.optimize_orch_tensors()(Before)

        call_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        dummy_calls = [name for name in call_names if name == "system.task_dummy"]
        assert "system.task_dummy" not in call_names, (
            f"Expected no cross-root task_dummy barriers, found: {dummy_calls}"
        )

    def test_submit_loop_writer_full_reader_uses_task_id_array_barrier(self):
        """Loop-local submit writers are collected into Array[TASK_ID].

        This is the indexer-like shape: the loop submits static output-window
        writes to ``score``, then a later full-parent reader consumes ``score``.
        The scalar writer tids are loop-local, so the pass must thread a
        TaskId array through the loop and use that array to feed a barrier
        before the full reader.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_next, _tid = pl.submit(self.writer, data, row, score)
                    result = self.reader(score, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        call_names: list[str] = []
        submit_count = 0
        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal submit_count, reader_dep_edges
                if isinstance(op.value, ir.Submit):
                    submit_count += 1
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        assert submit_count == 1
        assert "tensor.precompute_static_windows" in call_names
        assert "tensor.static_window_get" in call_names
        assert "array.create" in call_names
        assert "array.update_element" in call_names
        assert "system.task_dummy" in call_names
        assert reader_dep_edges == 1

    def test_parallel_submit_loop_writer_full_reader_uses_task_id_array_barrier(self):
        """Parallel loop-local window writer tids must also reach full readers.

        DeepSeek indexer uses a ``pl.parallel`` score writer followed by topk's
        full score read. The dependency planner must carry the per-lane TaskIds
        out through an Array[TASK_ID] exactly as it does for sequential loops.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi in pl.parallel(4):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_next, _tid = pl.submit(self.writer, data, row, score)
                    result = self.reader(score, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        call_names: list[str] = []
        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "array.create" in call_names
        assert "array.update_element" in call_names
        assert "system.task_dummy" in call_names
        assert reader_dep_edges == 1

    def test_two_loop_writers_same_parent_keep_separate_task_id_arrays(self):
        """Two loop writers to one parent must both feed the later full reader.

        A single per-parent array would overwrite slot ``bi`` when the second
        writer runs in the same iteration, leaving the reader dependent only on
        the second writer. Keep one TaskId array per writer record instead.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer_a(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 32], pl.FP32] = pl.load(data, [row_offset, 0], [64, 32])
                result: pl.Tile[[64, 32], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def writer_b(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 32], pl.FP32] = pl.load(data, [row_offset, 32], [64, 32])
                result: pl.Tile[[64, 32], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 32], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_a, _tid_a = pl.submit(self.writer_a, data, row, score)
                        _score_b, _tid_b = pl.submit(self.writer_b, data, row, score)
                    result = self.reader(score, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer_a__windowed") is not None
        assert After.get_function("writer_b__windowed") is not None

        array_create_count = 0
        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal array_create_count, reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "array.create"):
                    array_create_count += 1
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert array_create_count == 2
        assert reader_dep_edges == 1

    def test_scalar_submit_writer_full_reader_gets_dep(self):
        """A scalar (non-loop) submit writer + later full reader must take a dep.

        The writer tids are not loop-local, so the pass must drop the scalar
        tid directly into ``available_parent_deps_`` for the canonical root
        and the reader call's ``manual_dep_edges`` must reference it.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    _score_next, _tid = pl.submit(self.writer, data, 0, score)
                    result = self.reader(score, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        call_names: list[str] = []
        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        assert "tensor.precompute_static_windows" in call_names
        assert "tensor.static_window_get" in call_names
        # Scalar writer does NOT need an Array[TASK_ID] carry.
        assert "array.create" not in call_names
        assert "array.update_element" not in call_names
        assert "system.task_dummy" in call_names
        assert reader_dep_edges == 1

    def test_window_writer_full_reader_with_reshape_chain(self):
        """Reader passes score through ``tensor.reshape``; the alias chain must
        resolve to the writer's root so the dep is still inserted.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[128, 64], pl.FP32] = pl.load(data, [row_offset, 0], [128, 64])
                result: pl.Tile[[128, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score_flat: pl.Tensor[[16384], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[16384], pl.FP32] = pl.load(score_flat, [0], [16384])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                # 1D reshape of the 2D parent — same root, same storage.
                score_flat: pl.Tensor[[16384], pl.FP32] = pl.reshape(score, [16384])
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    _score_next, _tid = pl.submit(self.writer, data, 0, score)
                    result = self.reader(score_flat, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 1, "reshape chain must still resolve to writer root and insert dep"

    def test_loop_carried_parent_full_reader_gets_dep(self):
        """Loop-carried tensor return aliases must resolve back to the init root."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi, (score_iter,) in pl.range(4, init_values=(score,)):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        score_next, _tid = pl.submit(self.writer, data, row, score_iter)
                        score_final = pl.yield_(score_next)
                    result = self.reader(score_final, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 1

    def test_same_iteration_reader_gets_scalar_writer_dep(self):
        """A reader after a writer in the same loop iteration uses that tid."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_next, _tid = pl.submit(self.writer, data, row, score)
                        result = self.reader(score, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 1

    def test_loop_return_different_root_does_not_alias_init_root(self):
        """A loop return var must not alias its init root if yield changes root."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                other: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi, (score_iter,) in pl.range(4, init_values=(score,)):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_next, _tid = pl.submit(self.writer, data, row, score_iter)
                        score_final = pl.yield_(other)
                    result = self.reader(score_final, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 0

    def test_metadata_alias_full_reader_gets_dep(self):
        """Metadata-only tensor aliases should not hide a later full read."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                score_alias: pl.Tensor[[256, 64], pl.FP32] = pl.tensor.as_layout(
                    score, layout=pl.TensorLayout.ND
                )
                result: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    _score_next, _tid = pl.submit(self.writer, data, 0, score)
                    result = self.reader(score_alias, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_edges = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 1

    def test_window_writer_then_disjoint_window_reader_no_dep(self):
        """Two windows on the same parent that are provably disjoint must NOT
        take a barrier dep. The reader reads a different row block from the
        writer, so no TaskId edges should appear.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[128, 64], pl.FP32] = pl.load(data, [0, 0], [128, 64])
                result: pl.Tile[[128, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                block: pl.Tensor[[128, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile: pl.Tile[[128, 64], pl.FP32] = pl.load(block, [0, 0], [128, 64])
                ret: pl.Tensor[[128, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    # Writer covers rows [0, 128).
                    _score_next, _tid = pl.submit(self.writer, data, score)
                    # Reader covers rows [128, 256) — disjoint row block.
                    disjoint_block: pl.Tensor[[128, 64], pl.FP32] = pl.slice(score, [128, 64], [128, 0])
                    result = self.reader(disjoint_block, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None
        reader_dep_edges = 0
        task_dummy_count = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_edges, task_dummy_count
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader__windowed"):
                    reader_dep_edges += len((op.value.attrs or {}).get("manual_dep_edges", []))
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "system.task_dummy"):
                    task_dummy_count += 1
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_edges == 0, "disjoint row block must not take writer dep"
        # The windowed producer still emits its own task_dummy, but for the
        # reader side, no extra barrier should be inserted.
        assert task_dummy_count == 0, (
            f"disjoint window reader must not need a barrier, got {task_dummy_count}"
        )

    def test_three_root_writes_no_cross_deps(self):
        """q/k/v proj shape: three writers on three different storage roots
        must not serialize. Even when one of them is followed by a full
        reader, the dep must only land on the matching root.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[128, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[128, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                tensor: pl.Tensor[[128, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(tensor, [0, 0], [64, 64])
                ret: pl.Tensor[[128, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data_q: pl.Tensor[[128, 64], pl.FP32],
                data_k: pl.Tensor[[128, 64], pl.FP32],
                data_v: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                q_proj: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                k_proj: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                v_proj: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                result: pl.Tensor[[128, 64], pl.FP32] = pl.create_tensor([128, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    _q_next, _qt = pl.submit(self.writer, data_q, 0, q_proj)
                    _k_next, _kt = pl.submit(self.writer, data_k, 0, k_proj)
                    _v_next, _vt = pl.submit(self.writer, data_v, 0, v_proj)
                    # Full reader consumes only q_proj — only _qt should be a dep.
                    result = self.reader(q_proj, result)
                return result

        After = passes.optimize_orch_tensors()(Before)

        call_names: list[str] = []

        class _Collector(ir.IRVisitor):
            def visit_call(self, op):
                call_names.append(_call_name(op))
                super().visit_call(op)

        _Collector().visit_program(After)
        # No writer of one root appears in another root's reader deps.
        reader_dep_count = 0

        class _ReaderDepCounter(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_count
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader__windowed"):
                    reader_dep_count += len((op.value.attrs or {}).get("manual_dep_edges", []))
                super().visit_assign_stmt(op)

        _ReaderDepCounter().visit_program(After)
        # Only one writer matches q_proj's root → exactly one dep.
        assert reader_dep_count == 1, f"q-only reader should take exactly 1 dep, got {reader_dep_count}"
        # No barrier on k/v writers' roots (their tids are not consumed by q reader).
        # The single barrier is created with the q writer tid only.

    def test_unknown_alias_does_not_drop_dep(self):
        """If a reader's arg type is unknown to the region analysis, the planner
        must still conservatively insert a barrier (no false negative).
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile: pl.Tile[[32, 64], pl.FP32] = pl.load(data, [0, 0], [32, 64])
                result: pl.Tile[[32, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[64, 64], pl.FP32] = pl.store(result, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                src: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[32, 64], pl.FP32]],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                tile: pl.Tile[[32, 64], pl.FP32] = pl.load(src, [0, 0], [32, 64])
                ret: pl.Tensor[[32, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[32, 64], pl.FP32]:
                parent: pl.Tensor[[64, 64], pl.FP32] = pl.create_tensor([64, 64], dtype=pl.FP32)
                result: pl.Tensor[[32, 64], pl.FP32] = pl.create_tensor([32, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    _next, _tid = pl.submit(self.writer, data, parent)
                    # Reader gets a tensor variable with no alias chain in the
                    # planner's pass-local view. Resolver returns nullopt and
                    # planner conservatively inserts the dep.
                    result = self.reader(parent, result)
                return result

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        reader_dep_count = 0
        task_dummy_count = 0

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                nonlocal reader_dep_count, task_dummy_count
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "reader", "reader__windowed"):
                    reader_dep_count += len((op.value.attrs or {}).get("manual_dep_edges", []))
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "system.task_dummy"):
                    task_dummy_count += 1
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert reader_dep_count == 1
        assert task_dummy_count == 1

    def test_full_reader_summarizes_pending_writer_deps_for_later_readers(self):
        """A full reader compresses same-root pending writer deps into a barrier.

        Without root-level summary, each later full reader would rescan and
        depend on the whole writer TaskId array again. The first reader should
        consume the array barrier; the second reader should depend only on the
        first reader's barrier.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def writer(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
                row_offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [row_offset, 0], [64, 64])
                result: pl.Tile[[64, 64], pl.FP32] = pl.add(tile, tile)
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(result, [row_offset, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def reader(
                self,
                score: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                tile: pl.Tile[[64, 64], pl.FP32] = pl.load(score, [0, 0], [64, 64])
                ret: pl.Tensor[[256, 64], pl.FP32] = pl.store(tile, [0, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[256, 64], pl.FP32],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                score: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result_a: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                result_b: pl.Tensor[[256, 64], pl.FP32] = pl.create_tensor([256, 64], dtype=pl.FP32)
                with pl.manual_scope():
                    for bi in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = bi * 64
                        _score_next, _tid = pl.submit(self.writer, data, row, score)
                    result_a = self.reader(score, result_a)
                    result_b = self.reader(score, result_b)
                return result_b

        After = passes.optimize_orch_tensors()(Before)
        assert After.get_function("writer__windowed") is not None

        dummy_deps: list[list[str]] = []

        class _Collector(ir.IRVisitor):
            def visit_assign_stmt(self, op):
                if isinstance(op.value, ir.Call) and _is_call_named(op.value, "system.task_dummy"):
                    deps = (op.value.attrs or {}).get("manual_dep_edges", [])
                    dummy_deps.append([dep.name_hint for dep in deps])
                super().visit_assign_stmt(op)

        _Collector().visit_program(After)
        assert len(dummy_deps) == 2
        assert any("final" in dep for dep in dummy_deps[0]), dummy_deps
        assert len(dummy_deps[1]) == 1
        assert "window_barrier_tid" in dummy_deps[1][0], dummy_deps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
