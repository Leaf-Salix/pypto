# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime scope context managers for the PyPTO Language DSL."""


class manual_scope:
    """Context manager for a manual-dependency runtime scope.

    Inside this block, the runtime skips OverlapMap dependency tracking and
    TensorMap insert. The user declares every required ordering edge
    explicitly; the compiler lowers those declarations to TaskId deps:

      1. Tensor variables in ``deps=[var1, ...]`` are resolved to the TaskId
         companion of their producer.
      2. ``Scalar[TASK_ID]`` variables in ``deps=[task_id_var, ...]`` are
         passed through directly. ``task_id_var`` may be assigned from
         ``pl.task_id_of(result)``.
      3. Dependencies are supplied via the ``deps=[var1, task_id_var, ...]`` kwarg on any
         ``self.kernel(...)`` call inside this block. ``task_id_var`` may be
         assigned from ``pl.task_id_of(result)``.

    Usage::

        with pl.manual_scope():
            sij = self.qk_matmul(qi, kj, sij_buf)
            pij = self.softmax(sij)
            sij_tid = pl.task_id_of(sij)
            oi = self.pv_matmul(pij, vj, oi_buf,
                                deps=[pij, sij_tid])      # tensor + task-id deps

    Restrictions:
      - Must appear inside an Orchestration function (not InCore).
      - Cannot be nested inside another ``manual_scope`` (runtime forbids).
      - There is no automatic data-flow dependency inference inside this
        block; omit a required ``deps=[...]`` edge only when the kernels may
        run independently.
      - ``pl.no_dep(...)`` is rejected inside this block; use ``deps=[...]``
        to declare manual ordering.
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


__all__ = ["manual_scope"]
