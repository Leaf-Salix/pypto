# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the manual-scope lowering phase of DeriveCallDirections.

Manual-scope dep edge resolution and TaskId companion synthesis were originally
implemented as a standalone ``DeriveManualScopeDeps`` pass; they are now Phase 2
of ``DeriveCallDirections``. The tests here therefore drive the merged pass and
verify that the manual-scope behaviour is preserved.
"""

import pypto.language as pl
import pytest
from pypto import passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Skip the global print -> parse -> assert_structural_equal roundtrip.

    The python_printer does not surface ``Call.attrs['manual_dep_edges']`` (an
    internal post-pass attr), so the roundtrip would always fail after this
    pass. Property verification still runs.
    """
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


class TestManualScopeLoweringNoOp:
    def test_no_manual_scope_phase2_is_noop(self):
        """When no manual scope exists, Phase 2 of derive_call_directions is a no-op.

        Verified by idempotence: running the merged pass twice on the same input
        produces a structurally identical program. After the first run all
        ``arg_directions`` are populated and ``manual_dep_edges`` would only be
        written by Phase 2 — which has no manual scope to touch — so the second
        run returns the same Program.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        ssa = passes.convert_to_ssa()(Prog)
        first = passes.derive_call_directions()(ssa)
        second = passes.derive_call_directions()(first)
        assert second.same_as(first)


class TestManualScopeNesting:
    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_nested_manual_scope_rejected(self):
        """The runtime forbids MANUAL inside MANUAL; reject at parse time."""
        with pytest.raises(Exception, match="manual_scope"):  # noqa: B017

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        with pl.manual_scope():  # nested — must error
                            b = self.k1(a)
                    return b


class TestManualScopeTaskIdLowering:
    """Tests for ``Scalar[TASK_ID]`` passthrough in manual-scope lowering."""

    def test_taskid_var_passthrough_no_companion(self):
        """A ``Scalar[TASK_ID]`` dep must NOT get a ``__tid`` companion.

        The lowering should recognise that the var is already a TaskId and pass
        it through directly, not synthesise ``tid__ssa_v0__tid``.
        """
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    tid = pl.task_id_of(a)
                    b = self.k1(x, deps=[tid])
                return b

        ssa = passes.convert_to_ssa()(Prog)
        after = passes.derive_call_directions()(ssa)
        ir_dump = str(after)
        assert "tid__ssa_v0__tid" not in ir_dump

    def test_mixed_deps_selective_rewrite(self):
        """Tensor dep gets a companion; TaskId dep stays direct.

        ``deps=[a, tid]`` should produce:
        - ``a__ssa_v0__tid`` (companion for tensor dep)
        - NO companion for ``tid``
        """
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    b = self.k2(x)
                    tid = pl.task_id_of(b)
                    c = self.k1(x, deps=[a, tid])
                return c

        ssa = passes.convert_to_ssa()(Prog)
        after = passes.derive_call_directions()(ssa)
        ir_dump = str(after)
        assert "a__ssa_v0__tid" in ir_dump   # tensor companion
        assert "tid__ssa_v0__tid" not in ir_dump  # TaskId no companion

    def test_multiple_taskid_deps_passthrough(self):
        """Multiple TaskId deps all pass through directly."""
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                    b = self.k2(x)
                    tid_a = pl.task_id_of(a)
                    tid_b = pl.task_id_of(b)
                    c = self.k3(x, deps=[tid_a, tid_b])
                return c

        ssa = passes.convert_to_ssa()(Prog)
        after = passes.derive_call_directions()(ssa)
        ir_dump = str(after)
        assert "tid_a__ssa_v0__tid" not in ir_dump
        assert "tid_b__ssa_v0__tid" not in ir_dump

    def test_taskid_in_loop_body_no_companion(self):
        """TaskId inside a range loop body is not companion-lowered."""
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.range(4):
                        a = self.k1(x)
                        tid = pl.task_id_of(a)
                        x = self.k1(x, deps=[tid])
                return x

        ssa = passes.convert_to_ssa()(Prog)
        after = passes.derive_call_directions()(ssa)
        ir_dump = str(after)
        assert "tid__ssa_v0__tid" not in ir_dump


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
