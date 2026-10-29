# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure removing the early gate's unused Python Stream wrapper.

The Python raw-getter variant is diagnostic only. It keeps lazy initialization,
current device validation, both current-stream reads and the existing native
capture check. Production should expose this operation through PyPTO's adapter,
not depend on Torch's private Python getter. No frame/source/submit cache.
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from hbg_eager_latency import (
    FixedProbe,
    _DiagnosticPatches,
    add_kernel,
    configure_logging,
    measure,
    npu,
    torch,
    torch_npu,
)
from pypto.torch import launch


def check_raw_stream(config: Any) -> bool:
    from pypto.torch.interop import _load_torch_npu  # noqa: PLC0415

    module = _load_torch_npu()
    npu = module.npu
    device = npu.current_device()
    if device != config.device_id:
        raise ValueError(f"Kernel call device {device} differs from initialized NPU {config.device_id}")
    npu._lazy_init()
    stream_id, _, _ = module._C._npu_getCurrentStream(device)
    return launch._check_capture(stream_id, device)


def live_checks(probe: FixedProbe, candidate: Any) -> dict[str, Any]:
    """Keep the frame's post-resolver stream read and error ordering observable."""
    patches = _DiagnosticPatches()
    original = npu.current_stream()
    alternate = npu.Stream(device=probe.device)
    resolver = add_kernel._resolve_compiled
    enqueue = probe.native.enqueue
    observed = []

    def resolve(*args: Any, **kwargs: Any) -> Any:
        result = resolver(*args, **kwargs)
        npu.set_stream(alternate)
        return result

    def record(tensors: Any, stream_id: int, *args: Any) -> Any:
        observed.append(stream_id)
        return enqueue(tensors, stream_id, *args)

    probe.out.fill_(float("nan"))
    npu.synchronize()
    try:
        assert candidate(probe.config) is False
        assert launch.check_eager_call(probe.config) is False
        patches.set(launch, "check_eager_call", candidate)
        patches.set(add_kernel, "_resolve_compiled", resolve)
        patches.set(probe.native, "enqueue", record)
        assert add_kernel(*probe.inputs) is probe.out
        alternate.synchronize()
        probe.check()
        assert observed == [alternate.stream_id]
    finally:
        patches.restore()
        npu.set_stream(original)
        probe.state.drain()

    # Wrong-device rejection must precede the native capture operation.
    def unexpected(*args: Any) -> Any:
        raise AssertionError("device mismatch reached capture check")

    try:
        patches.set(launch, "_check_capture", unexpected)
        try:
            candidate(replace(probe.config, device_id=probe.device + 1))
        except ValueError as error:
            assert "differs from initialized" in str(error)
        else:
            raise AssertionError("device mismatch was accepted")
    finally:
        patches.restore()
    return {
        "resolver_stream_switch": "frame/enqueue uses the new stream",
        "device_mismatch": "before capture",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--baseline-launch", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.batches <= 0:
        parser.error("samples and batches must be positive")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT"):
        parser.error("run without native instrumentation")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "identity.json").write_text(
        json.dumps(
            {
                "pypto": importlib.import_module("pypto").__file__,
                "simpler": importlib.import_module("simpler").__file__,
                "torch_npu": torch_npu.__file__,
                "torch_version": torch.__version__,
                "device": args.device,
                "queue": os.environ.get("TASK_QUEUE_ENABLE"),
                "candidate": (
                    "native current-stream capture check"
                    if args.baseline_launch
                    else "raw getter with original loader, device and capture checks"
                ),
            },
            indent=2,
        )
    )
    baseline_helpers: dict[str, Any] = {}
    baseline = launch.check_eager_call
    candidate = check_raw_stream
    candidate_name = "raw_gate"
    if args.baseline_launch is not None:
        source = args.baseline_launch.read_text()
        tree = ast.parse(source)
        names = {"_load_native", "_check_capture", "check_eager_call"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in nodes} == names
        local: dict[str, Any] = {}
        exec(
            compile(
                ast.Module(body=cast(list[ast.stmt], nodes), type_ignores=[]), "<baseline-launch>", "exec"
            ),
            cast(dict[str, Any], vars(launch)),
            local,
        )
        baseline_helpers = {name: local[name] for name in names}
        baseline = local["check_eager_call"]
        candidate = launch.check_eager_call
        candidate_name = "native_gate"
        (args.output / "baseline.sha256").write_text(hashlib.sha256(source.encode()).hexdigest() + "\n")
    configure_logging("error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    probe = FixedProbe(args.device)
    patches = _DiagnosticPatches()
    prepare, finish = probe.prepare, probe.finish

    def prepare_variant(stage: str) -> None:
        patches.restore()
        prepare("aclnn" if stage == "aclnn" else "jit")
        (probe.aclnn_out if stage == "aclnn" else probe.out).fill_(float("nan"))
        npu.synchronize()
        if stage == "baseline":
            for name, helper in baseline_helpers.items():
                patches.set(launch, name, helper)
        if stage != "aclnn":
            patches.set(launch, "check_eager_call", baseline if stage == "baseline" else candidate)

    def finish_variant() -> None:
        try:
            finish()
        finally:
            patches.restore()

    try:
        (args.output / "live-checks.json").write_text(json.dumps(live_checks(probe, candidate), indent=2))
        probe.prepare, probe.finish = prepare_variant, finish_variant
        measure(probe, args.output, args.samples, args.batches, ("baseline", candidate_name, "aclnn"))
    finally:
        try:
            finish_variant()
        finally:
            probe.prepare, probe.finish = prepare, finish
            probe.state.close()


if __name__ == "__main__":
    main()
