# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser tests for ``with pl.manual_scope():`` and the ``deps=[var]`` Call kwarg."""

import pypto.language as pl
import pytest
from pypto import ir


def _first_runtime_scope(stmt):
    """Return the first RuntimeScopeStmt found in a stmt subtree (DFS), or None."""
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _first_runtime_scope(s)
            if r is not None:
                return r
    return None


class TestManualScopeParsing:
    def test_parse_manual_scope_creates_runtime_scope_with_manual_true(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a = self.k1(x)
                return a

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None, "expected a RuntimeScopeStmt for `with pl.manual_scope():`"
        assert scope.manual is True

    def test_parse_manual_scope_rejects_arguments(self):
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope(name="foo"):
                        return x

    def test_deps_kwarg_records_user_manual_dep_edges(self):
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
                    b = self.k2(x, deps=[a])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        # Walk the scope body and inspect both Calls.
        body = scope.body
        if isinstance(body, ir.SeqStmts):
            stmts = list(body.stmts)
        else:
            stmts = [body]
        assert len(stmts) == 2
        a_assign, b_assign = stmts
        assert isinstance(a_assign, ir.AssignStmt)
        assert isinstance(b_assign, ir.AssignStmt)
        a_call = a_assign.value
        b_call = b_assign.value
        assert isinstance(a_call, ir.Call)
        assert isinstance(b_call, ir.Call)
        assert "user_manual_dep_edges" not in a_call.attrs
        b_user_deps = b_call.attrs.get("user_manual_dep_edges", [])
        assert len(b_user_deps) == 1
        assert b_user_deps[0].same_as(a_assign.var)

    def test_deps_kwarg_accepts_task_id_var(self):
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
                    tid = pl.task_id_of(a)
                    b = self.k2(x, deps=[tid])
                return b

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        body = scope.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        assert len(stmts) == 3
        _, tid_assign, b_assign = stmts
        assert isinstance(tid_assign, ir.AssignStmt)
        assert isinstance(b_assign, ir.AssignStmt)
        assert isinstance(tid_assign.value, ir.Call)
        assert tid_assign.value.op.name == "system.task_id_of"
        assert isinstance(tid_assign.var.type, ir.ScalarType)
        assert tid_assign.var.type.dtype == pl.TASK_ID
        b_call = b_assign.value
        assert isinstance(b_call, ir.Call)
        b_user_deps = b_call.attrs.get("user_manual_dep_edges", [])
        assert len(b_user_deps) == 1
        assert b_user_deps[0].same_as(tid_assign.var)

    def test_deps_kwarg_accepts_mixed_tensor_and_task_id_vars(self):
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
                    tid = pl.task_id_of(b)
                    c = self.k3(x, deps=[a, tid])
                return c

        fn = Prog.get_function("main")
        assert fn is not None
        scope = _first_runtime_scope(fn.body)
        assert scope is not None
        body = scope.body
        stmts = list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]
        assert len(stmts) == 4
        a_assign, _, tid_assign, c_assign = stmts
        assert isinstance(a_assign, ir.AssignStmt)
        assert isinstance(tid_assign, ir.AssignStmt)
        assert isinstance(c_assign, ir.AssignStmt)
        c_call = c_assign.value
        assert isinstance(c_call, ir.Call)
        c_user_deps = c_call.attrs.get("user_manual_dep_edges", [])
        assert len(c_user_deps) == 2
        assert c_user_deps[0].same_as(a_assign.var)
        assert c_user_deps[1].same_as(tid_assign.var)

    def test_deps_kwarg_rejects_inline_task_id_of(self):
        with pytest.raises(Exception):  # noqa: B017 - parser raises ParserSyntaxError/ParserTypeError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        b = self.k1(x, deps=[pl.task_id_of(a)])
                    return b

    def test_manual_scope_rejects_no_dep_wrapper(self):
        with pytest.raises(Exception):  # noqa: B017 - parser raises ParserSyntaxError/ParserTypeError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(pl.no_dep(x))
                    return a

    def test_deps_outside_manual_scope_is_rejected(self):
        with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    a = self.k1(x)
                    # No manual_scope around this — the deps= kwarg must error.
                    b = self.k1(x, deps=[a])
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
