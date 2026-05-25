# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""pypto-lib inspired ST witnesses for manual phase-fence compression.

The source pypto-lib models are intentionally not imported here: these tests are
self-contained mini workloads that preserve the real phase topology and names,
then expose the dependency pattern with ``manual_scope`` + ``deps=[tids]``.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output"

_BRANCHES = 4
_ROWS_PER_BRANCH = 16
_COLS = 32
_ROWS = _BRANCHES * _ROWS_PER_BRANCH
_SCALE_EPS = 1e-4
_PERF_ENV = "PYPTO_LIB_PHASE_FENCE_PERF"
_PERF_BRANCHES = 64
_PERF_ROWS_PER_BRANCH = 1
_PERF_COLS = 32
_PERF_ROWS = _PERF_BRANCHES * _PERF_ROWS_PER_BRANCH


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    program_builder: Callable[[], type]
    expected_builder: Callable[[torch.Tensor], torch.Tensor]
    producer_hint: str
    consumer_hint: str


def _prefill_expected(data: torch.Tensor) -> torch.Tensor:
    scratch = torch.abs(data) + 0.25
    scales = torch.clamp(127.0 / (scratch + _SCALE_EPS), max=127.0)
    return torch.round(torch.clamp(data * scales, min=-127.0, max=127.0))


def _decode_expected(data: torch.Tensor) -> torch.Tensor:
    scratch = data * 1.5 + 0.125
    return torch.softmax(scratch, dim=1)


def _qkv_rope_expected(data: torch.Tensor) -> torch.Tensor:
    scratch = data * 0.75 + 0.5
    return torch.round(torch.clamp(scratch * 16.0, min=-127.0, max=127.0))


def _build_prefill_sparse_attn_phase_fence_program():
    """Mini of pypto-lib deepseek/v4/prefill_sparse_attn.py Stage 5 -> 6."""

    @pl.program
    class DeepSeekPrefillSparseAttnPhaseFence:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            scratch: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
        ) -> pl.Tensor[[_ROWS, _COLS], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(_BRANCHES, pl.TASK_ID)
                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="prefill_stage_a_store_amax_tile",
                    ) as tid:
                        tile = data[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        abs_tile = pl.maximum(tile, pl.neg(tile))
                        scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS] = pl.add(abs_tile, 0.25)
                    tids[branch] = tid

                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="prefill_stage_b_quant_tile",
                        deps=[tids],
                    ):
                        tile = data[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        amax = scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        scale = pl.div(
                            pl.full([_ROWS_PER_BRANCH, _COLS], dtype=pl.FP32, value=127.0),
                            pl.add(amax, _SCALE_EPS),
                        )
                        clamped = pl.minimum(scale, 127.0)
                        quant_i32 = pl.cast(pl.mul(tile, clamped), target_type=pl.INT32, mode="rint")
                        quant = pl.cast(quant_i32, target_type=pl.FP32, mode="none")
                        out[row : row + _ROWS_PER_BRANCH, 0:_COLS] = quant
            return out

    return DeepSeekPrefillSparseAttnPhaseFence


def _build_decode_sparse_attn_phase_fence_program():
    """Mini of pypto-lib deepseek/v4/decode_sparse_attn.py gather -> QK/softmax."""

    @pl.program
    class DeepSeekDecodeSparseAttnPhaseFence:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            scratch: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
        ) -> pl.Tensor[[_ROWS, _COLS], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(_BRANCHES, pl.TASK_ID)
                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="cfa_proj_gather_kv_topk_tile",
                    ) as tid:
                        tile = data[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS] = pl.add(pl.mul(tile, 1.5), 0.125)
                    tids[branch] = tid

                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="cfa_proj_sparse_attn_qk_softmax_tile",
                        deps=[tids],
                    ):
                        gathered = scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        max_v = pl.row_max(gathered)
                        exp_v = pl.exp(pl.row_expand_sub(gathered, max_v))
                        denom = pl.row_sum(exp_v)
                        out[row : row + _ROWS_PER_BRANCH, 0:_COLS] = pl.row_expand_div(exp_v, denom)
            return out

    return DeepSeekDecodeSparseAttnPhaseFence


def _build_qkv_proj_rope_phase_fence_program():
    """Mini of pypto-lib deepseek/v4/qkv_proj_rope.py qr_norm_apply -> qr_quant_apply."""

    @pl.program
    class DeepSeekQkvProjRopePhaseFence:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            scratch: pl.Tensor[[_ROWS, _COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[_ROWS, _COLS], pl.FP32]],
        ) -> pl.Tensor[[_ROWS, _COLS], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(_BRANCHES, pl.TASK_ID)
                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_norm_apply") as tid:
                        tile = data[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS] = pl.add(pl.mul(tile, 0.75), 0.5)
                    tids[branch] = tid

                for branch in pl.parallel(_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="qr_quant_apply",
                        deps=[tids],
                    ):
                        normed = scratch[row : row + _ROWS_PER_BRANCH, 0:_COLS]
                        scaled = pl.mul(normed, 16.0)
                        clipped = pl.maximum(pl.minimum(scaled, 127.0), -127.0)
                        quant_i32 = pl.cast(clipped, target_type=pl.INT32, mode="rint")
                        out[row : row + _ROWS_PER_BRANCH, 0:_COLS] = pl.cast(
                            quant_i32,
                            target_type=pl.FP32,
                            mode="none",
                        )
            return out

    return DeepSeekQkvProjRopePhaseFence


def _build_prefill_sparse_attn_phase_fence_perf64_program():
    """64x64 manual profiling witness: prefill Stage-A producers -> Stage-B consumers."""

    @pl.program
    class DeepSeekPrefillSparseAttnPhaseFencePerf64:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            data: pl.Tensor[[_PERF_ROWS, _PERF_COLS], pl.FP32],
            scratch: pl.Tensor[[_PERF_ROWS, _PERF_COLS], pl.FP32],
            out: pl.Out[pl.Tensor[[_PERF_ROWS, _PERF_COLS], pl.FP32]],
        ) -> pl.Tensor[[_PERF_ROWS, _PERF_COLS], pl.FP32]:
            with pl.manual_scope():
                tids = pl.array.create(_PERF_BRANCHES, pl.TASK_ID)
                for branch in pl.parallel(_PERF_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _PERF_ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="prefill_perf64_stage_a_store_amax_tile",
                    ) as tid:
                        tile = data[row : row + _PERF_ROWS_PER_BRANCH, 0:_PERF_COLS]
                        scratch[row : row + _PERF_ROWS_PER_BRANCH, 0:_PERF_COLS] = pl.add(tile, 1.0)
                    tids[branch] = tid

                for branch in pl.parallel(_PERF_BRANCHES):
                    row: pl.Scalar[pl.INDEX] = branch * _PERF_ROWS_PER_BRANCH
                    with pl.at(
                        level=pl.Level.CORE_GROUP,
                        name_hint="prefill_perf64_stage_b_quant_tile",
                        deps=[tids],
                    ):
                        tile = scratch[row : row + _PERF_ROWS_PER_BRANCH, 0:_PERF_COLS]
                        out[row : row + _PERF_ROWS_PER_BRANCH, 0:_PERF_COLS] = pl.mul(tile, 2.0)
            return out

    return DeepSeekPrefillSparseAttnPhaseFencePerf64


_CASE_SPECS = [
    _CaseSpec(
        name="pypto_lib_deepseek_prefill_sparse_attn_phase_fence",
        program_builder=_build_prefill_sparse_attn_phase_fence_program,
        expected_builder=_prefill_expected,
        producer_hint="prefill_stage_a_store_amax_tile",
        consumer_hint="prefill_stage_b_quant_tile",
    ),
    _CaseSpec(
        name="pypto_lib_deepseek_decode_sparse_attn_phase_fence",
        program_builder=_build_decode_sparse_attn_phase_fence_program,
        expected_builder=_decode_expected,
        producer_hint="cfa_proj_gather_kv_topk_tile",
        consumer_hint="cfa_proj_sparse_attn_qk_softmax_tile",
    ),
    _CaseSpec(
        name="pypto_lib_deepseek_qkv_proj_rope_phase_fence",
        program_builder=_build_qkv_proj_rope_phase_fence_program,
        expected_builder=_qkv_rope_expected,
        producer_hint="qr_norm_apply",
        consumer_hint="qr_quant_apply",
    ),
]


def _input_tensor() -> torch.Tensor:
    values = torch.arange(_ROWS * _COLS, dtype=torch.float32).reshape(_ROWS, _COLS)
    return values / 257.0 - 4.0


def _perf_input_tensor() -> torch.Tensor:
    values = torch.arange(_PERF_ROWS * _PERF_COLS, dtype=torch.float32).reshape(_PERF_ROWS, _PERF_COLS)
    return values / 257.0 - 4.0


class _PyptoLibPhaseFenceCase(PTOTestCase):
    __test__ = False

    def __init__(self, spec: _CaseSpec, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)
        self._spec = spec

    def get_name(self) -> str:
        return self._spec.name

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [_ROWS, _COLS], DataType.FP32, init_value=_input_tensor),
            TensorSpec("scratch", [_ROWS, _COLS], DataType.FP32, init_value=0.0),
            TensorSpec("out", [_ROWS, _COLS], DataType.FP32, init_value=0.0, is_output=True),
        ]

    def get_program(self) -> Any:
        return self._spec.program_builder()

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = self._spec.expected_builder(tensors["data"])


def _case(spec: _CaseSpec, *, platform: str | None = None) -> _PyptoLibPhaseFenceCase:
    return _PyptoLibPhaseFenceCase(spec, platform=platform)


class _PyptoLibPhaseFencePerf64Case(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform: str | None = None, config=None):
        super().__init__(config, platform=platform)

    def get_name(self) -> str:
        return "pypto_lib_deepseek_prefill_sparse_attn_phase_fence_perf64x64"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [_PERF_ROWS, _PERF_COLS], DataType.FP32, init_value=_perf_input_tensor),
            TensorSpec("scratch", [_PERF_ROWS, _PERF_COLS], DataType.FP32, init_value=0.0),
            TensorSpec("out", [_PERF_ROWS, _PERF_COLS], DataType.FP32, init_value=0.0, is_output=True),
        ]

    def get_program(self) -> Any:
        return _build_prefill_sparse_attn_phase_fence_perf64_program()

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = (tensors["data"] + 1.0) * 2.0


def _new_swimlane_json(test_runner, case: PTOTestCase, *, label: str) -> dict:
    if not test_runner.config.enable_l2_swimlane:
        pytest.skip(f"pass --enable-l2-swimlane to validate {label}")
    before: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    result = test_runner.run(case)
    assert result.passed, f"{label} failed: {result.error}"
    after: set[Path] = set(_BUILD_OUTPUT_DIR.glob("*/dfx_outputs/l2_perf_records.json"))
    candidates = list(after - before)
    if not candidates:
        candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    assert candidates, f"No l2_perf_records.json generated for {label}"
    return json.loads(max(candidates, key=lambda p: p.stat().st_mtime).read_text())


def _assert_two_phase_strict(swimlane_data: dict, *, label: str, branches: int = _BRANCHES) -> None:
    expected = 2 * branches
    tasks = swimlane_data["tasks"]
    if len(tasks) < expected:
        pytest.skip(f"need >= {expected} tasks for {label}, got {len(tasks)}")
    tasks = sorted(tasks, key=lambda t: t["start_time_us"])[:expected]
    producers = tasks[:branches]
    consumers = tasks[branches:expected]
    producer_end = max(t["end_time_us"] for t in producers)
    consumer_start = min(t["start_time_us"] for t in consumers)
    assert consumer_start >= producer_end, (
        f"{label} consumer phase starts at {consumer_start:.2f}us before "
        f"producer phase ends at {producer_end:.2f}us"
    )


def _generate_orch_code(program_cls) -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    transformed = passes.derive_call_directions()(transformed)
    transformed = passes.expand_manual_phase_fence()(transformed)
    for func in transformed.functions.values():
        if func.func_type == ir.FunctionType.Orchestration:
            return codegen.generate_orchestration(transformed, func).code
    raise ValueError("No orchestration function found in program")


def _assert_compressed_barrier_codegen(code: str) -> None:
    assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
    assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{_BRANCHES}];" in code, code
    real_dep_arrays = re.findall(r"PTO2TaskId (params_t\d+)_deps\[1\];", code)
    assert real_dep_arrays, code
    assert not re.search(rf"PTO2TaskId params_t\d+_deps\[{_BRANCHES}\];", code), code


class TestPyptoLibPhaseFenceCorrectness:
    @pytest.fixture(autouse=True)
    def _skip_when_collecting_l2_swimlane(self, test_runner):
        if test_runner.config.enable_l2_swimlane:
            pytest.skip("correctness cases run without --enable-l2-swimlane")

    @pytest.mark.parametrize("spec", _CASE_SPECS, ids=[spec.name for spec in _CASE_SPECS])
    @pytest.mark.parametrize("platform", PLATFORMS)
    def test_correctness(self, test_runner, spec: _CaseSpec, platform: str):
        result = test_runner.run(_case(spec, platform=platform))
        assert result.passed, f"{spec.name} failed: {result.error}"


class TestPyptoLibPhaseFenceSwimlane:
    @pytest.mark.parametrize("spec", _CASE_SPECS, ids=[spec.name for spec in _CASE_SPECS])
    def test_phase_strictness(self, test_runner, spec: _CaseSpec):
        data = _new_swimlane_json(test_runner, _case(spec), label=spec.name)
        _assert_two_phase_strict(data, label=spec.name)


class TestPyptoLibPhaseFencePerfSwimlane:
    def test_prefill_perf64x64_phase_strictness(self, test_runner):
        if os.environ.get(_PERF_ENV) != "1":
            pytest.skip(f"set {_PERF_ENV}=1 to run the 64x64 manual profiling witness")
        label = "pypto_lib_deepseek_prefill_sparse_attn_phase_fence_perf64x64"
        data = _new_swimlane_json(test_runner, _PyptoLibPhaseFencePerf64Case(), label=label)
        _assert_two_phase_strict(data, label=label, branches=_PERF_BRANCHES)


class TestPyptoLibPhaseFenceCodegen:
    @pytest.mark.parametrize("spec", _CASE_SPECS, ids=[spec.name for spec in _CASE_SPECS])
    def test_codegen_uses_compressed_dummy_barrier(self, spec: _CaseSpec):
        code = _generate_orch_code(spec.program_builder())
        _assert_compressed_barrier_codegen(code)
        assert spec.producer_hint in code
        assert spec.consumer_hint in code
