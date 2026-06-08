# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reproduce the DeepSeek-V4 decode indexer score -> topk dependency bug.

The copied model files come from pypto-lib commit
8ee6911f5f2c8737489a8d0858d17e6945ff948f.  This test intentionally runs that
old decode indexer against the local PyPTO checkout so that a pass-pipeline
change that windows score writes but still allows a later full-score topk read
can be validated on hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_CASE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _CASE_DIR / "models" / "deepseek" / "v4"
for _path in (_CASE_DIR, _MODEL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from golden import ratio_allclose, run_jit, topk_pair_compare  # noqa: E402
from indexer import (  # noqa: E402
    IDX_KV_LEN,
    IDX_TOPK,
    OFFSET,
    build_tensor_specs,
    golden_indexer,
    indexer_test,
)


def _topk_idxs_compare(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
    """Compare topk idxs against the final score tensor, tolerating legal ties."""
    score = actual_outputs["score"].cpu().to(torch.float32)
    actual_topk = actual[..., :IDX_TOPK]
    expected_topk = expected[..., :IDX_TOPK]
    actual_score_index = (actual_topk.long() - OFFSET).clamp(min=0, max=score.shape[-1] - 1)
    paired_scores = torch.gather(score, dim=-1, index=actual_score_index)
    synthetic_outputs = {**actual_outputs, "_paired_topk_scores": paired_scores}
    return topk_pair_compare("_paired_topk_scores")(
        actual_topk,
        expected_topk,
        actual_outputs=synthetic_outputs,
        expected_outputs=expected_outputs,
        inputs=inputs,
        rtol=rtol,
        atol=atol,
    )


_topk_idxs_compare.__name__ = "topk_pair_compare"


@pytest.mark.platforms("a2a3")
def test_decode_indexer_score_to_topk_guard_repro(request):
    torch.manual_seed(20260608)
    device_opt = str(request.config.getoption("--device"))
    device_id = int(device_opt.split(",", maxsplit=1)[0].split("-", maxsplit=1)[0])
    result = run_jit(
        fn=indexer_test,
        specs=build_tensor_specs(),
        golden_fn=golden_indexer,
        runtime_cfg=dict(
            platform=request.config.getoption("--platform"),
            device_id=device_id,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "score": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "topk_idxs": _topk_idxs_compare,
            "idx_kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005 / IDX_KV_LEN),
        },
    )
    assert result.passed, result.error
