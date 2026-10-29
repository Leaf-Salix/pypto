# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Randomized, drained ablations of the complete HBG eager path.

Run each queue setting and seed in a fresh process. Historical functions restore
selected work using current module globals; no timed wrappers, cache clearing,
or guard bypasses are used. Deltas are conditional on all other current changes.
The two current labels are identical implementations and calibrate block noise.
"""

import argparse
import ast
import hashlib
import importlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from hbg_eager_latency import FixedProbe, _DiagnosticPatches, add_kernel, configure_logging, npu, torch

STAGES = (
    "current_a",
    "current_b",
    "off_session",
    "off_structure",
    "off_resolver",
    "off_initial_frame",
    "off_import",
    "off_source",
    "aclnn",
)


class Ablations:
    """Install exact function bodies only between fully drained blocks."""

    def __init__(self, baseline_root: Path) -> None:
        self.root = baseline_root
        self.patches = _DiagnosticPatches()
        self.hashes: dict[str, str] = {}
        self.plans: dict[str, list[tuple[Any, str, Any, Any]]] = {stage: [] for stage in STAGES}
        self.modules = {
            name: importlib.import_module(name)
            for name in (
                "pypto.jit._arguments",
                "pypto.jit._ast_analysis",
                "pypto.jit._call_binding",
                "pypto.jit._kernel",
                "pypto.jit.decorator",
                "pypto.torch.interop",
                "pypto.torch.launch",
                "pypto.runtime.kernel.context",
                "pypto.runtime.kernel.session",
            )
        }
        for module in self.modules.values():
            assert module.__file__ is not None
            path = Path(module.__file__)
            self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.hashes[__file__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        jit = self.modules["pypto.jit.decorator"]
        session = self.modules["pypto.runtime.kernel.session"]
        context = self.modules["pypto.runtime.kernel.context"]
        self.add("off_initial_frame", "88761d3^", "pypto.jit._kernel", "invoke_jit")
        self.add(
            "off_session",
            "b1533ec^",
            session.__name__,
            "device_location",
            session._BorrowedWorkerSession,
            "_BorrowedWorkerSession",
        )
        self.add("off_session", None, "pypto.torch.launch", "invoke", remove_keyword="api")
        self.add("off_structure", "e8b14a7^", "pypto.jit._ast_analysis", "constant_dependency_names")
        for name in ("_get_source_hash", "_resolve_constexpr_bindings"):
            self.add("off_structure", "e8b14a7^", jit.__name__, name, jit.JITFunction, "JITFunction")
        self.add("off_structure", "e8b14a7^", "pypto.jit._call_binding", "expand_constexpr_variants")
        self.add("off_structure", "e8b14a7^", "pypto.jit._call_binding", "_dep_call_nodes")
        for name in ("_annotation_namespace", "param_layouts", "param_dtypes", "constexpr_params"):
            self.add("off_resolver", "803d45f^", "pypto.jit._arguments", name)
        for name in ("_param_names", "_bind_args"):
            self.add("off_resolver", "803d45f^", jit.__name__, name, jit.JITFunction, "JITFunction")
        self.add("off_import", "2bef482^", "pypto.torch.launch", "_load_native")
        self.add("off_import", "2bef482^", "pypto.torch.interop", "_load_torch_npu")
        self.add("off_source", None, "pypto.torch.launch", "invoke", remove_keyword="hold_sources")
        self.add(
            "off_source",
            "803d45f^",
            context.__name__,
            "submit",
            context._ProcessKernelState,
            "_ProcessKernelState",
        )

    def add(
        self,
        stage: str,
        revision: str | None,
        module_name: str,
        name: str,
        owner: Any = None,
        class_name: str | None = None,
        remove_keyword: str | None = None,
    ) -> None:
        module = self.modules[module_name]
        relative = Path("python") / Path(*module_name.split(".")).with_suffix(".py")
        assert module.__file__ is not None
        path = self.root / revision / relative if revision else Path(module.__file__)
        source = path.read_text()
        self.hashes[str(path)] = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source, filename=str(path))
        body = tree.body
        if class_name:
            classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]
            assert len(classes) == 1, (path, class_name)
            body = classes[0].body
        nodes = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(nodes) == 1, (path, name)
        node = nodes[0]
        if remove_keyword:
            removed = 0
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    removed += sum(keyword.arg == remove_keyword for keyword in call.keywords)
                    call.keywords = [keyword for keyword in call.keywords if keyword.arg != remove_keyword]
            assert removed == 1, (stage, remove_keyword, removed)
        statements: list[ast.stmt] = [node]
        code = compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec")
        self.plans[stage].append((owner if owner is not None else module, name, module, code))

    def set(self, owner: Any, name: str, value: Any) -> None:
        # _annotation_namespace and importlib do not exist in current modules.
        if not hasattr(owner, name):
            self.patches.saved.append((owner, name, False, None))
            setattr(owner, name, value)
        else:
            self.patches.set(owner, name, value)

    def install(self, stage: str) -> None:
        assert not self.patches.saved
        if stage == "off_import":
            self.set(self.modules["pypto.torch.launch"], "importlib", importlib)
        for owner, name, module, code in self.plans[stage]:
            namespace = vars(module)
            existed, saved = name in namespace, namespace.get(name)
            # Compile-time decorators use the real globals, as do later calls.
            try:
                exec(code, namespace)
                value = namespace[name]
            finally:
                if existed:
                    namespace[name] = saved
                else:
                    namespace.pop(name, None)
            self.set(owner, name, value)

    def restore(self) -> None:
        self.patches.restore()


def measure(probe: FixedProbe, ablations: Ablations, args: Any) -> None:
    rng = random.Random(args.seed)
    host, lifecycle, orders = [], [], []

    def prepare(stage: str) -> None:
        probe.finish()
        ablations.restore()
        ablations.install(stage)
        probe.prepare("aclnn" if stage == "aclnn" else "jit")
        # Same allocation contract, changing values; stale output cannot pass.
        probe.x.add_(1)
        probe.expected.add_(1)
        target = probe.aclnn_out if stage == "aclnn" else probe.out
        target.fill_(float("nan"))
        npu.synchronize()

    def check(stage: str) -> None:
        target = probe.aclnn_out if stage == "aclnn" else probe.out
        torch.testing.assert_close(target.cpu(), probe.expected, rtol=0, atol=0)

    def finish() -> None:
        try:
            probe.finish()
        finally:
            ablations.restore()

    for batch in range(args.batches):
        order = list(STAGES)
        rng.shuffle(order)
        orders.append({"kind": "host", "batch": batch, "order": order})
        for stage in order:
            prepare(stage)
            try:
                for _ in range(50):
                    probe.call()
                    npu.synchronize()
                for sample in range(args.samples):
                    npu.synchronize()
                    start = time.perf_counter_ns()
                    cpu = time.thread_time_ns()
                    probe.call()
                    cpu_end = time.thread_time_ns()
                    end = time.perf_counter_ns()
                    npu.synchronize()
                    host.append(
                        {
                            "batch": batch,
                            "stage": stage,
                            "sample": sample,
                            "host_us": (end - start) / 1000,
                            "caller_cpu_us": (cpu_end - cpu) / 1000,
                        }
                    )
                check(stage)
            finally:
                finish()
            print("HOST", batch, stage, flush=True)
        (args.output / "host.json").write_text(json.dumps(host))
        (args.output / "orders.json").write_text(json.dumps(orders))
    for batch in range(args.batches):
        order = list(STAGES)
        rng.shuffle(order)
        orders.append({"kind": "lifecycle", "batch": batch, "order": order})
        for stage in order:
            # Installing the counterfactual and poisoning outputs are diagnostic
            # machinery, excluded from lifecycle. Preparation and true release
            # of each measured group remain included.
            prepare(stage)
            probe.finish()
            start = time.perf_counter_ns()
            cpu = time.process_time_ns()
            try:
                probe.prepare("aclnn" if stage == "aclnn" else "jit")
                for _ in range(100):
                    probe.call()
                    npu.synchronize()
            finally:
                probe.finish()
            cpu_end = time.process_time_ns()
            end = time.perf_counter_ns()
            try:
                check(stage)
                lifecycle.append(
                    {
                        "batch": batch,
                        "stage": stage,
                        "cpu_us": (cpu_end - cpu) / 100000,
                        "wall_us": (end - start) / 100000,
                    }
                )
            finally:
                ablations.restore()
            print("CPU", batch, stage, flush=True)
        (args.output / "lifecycle.json").write_text(json.dumps(lifecycle))
        (args.output / "orders.json").write_text(json.dumps(orders))
    summary = {
        stage: {
            "host_us": statistics.median(row["host_us"] for row in host if row["stage"] == stage),
            "lifecycle_cpu_us": statistics.median(
                row["cpu_us"] for row in lifecycle if row["stage"] == stage
            ),
        }
        for stage in STAGES
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batches", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.batches < 1:
        parser.error("samples and batches must be positive")
    if os.environ.get("LD_PRELOAD") or os.environ.get("PYPTO_LAUNCH_PROBE_OUTPUT"):
        parser.error("formal timings require no preload or launch instrumentation")
    args.output.mkdir(parents=True, exist_ok=True)
    ablations = Ablations(args.baseline_root)
    configure_logging("error")
    torch.set_num_threads(1)
    npu.set_device(args.device)
    npu.set_compile_mode(jit_compile=False)
    identity = {
        "seed": args.seed,
        "device": args.device,
        "queue": os.environ.get("TASK_QUEUE_ENABLE"),
        "samples": args.samples,
        "batches": args.batches,
        "source_hashes": ablations.hashes,
        "imports": {name: module.__file__ for name, module in ablations.modules.items()},
        "scope": "conditional full-JIT ablations; no independent request-signature claim",
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))
    probe = FixedProbe(args.device)

    def unexpected_compile(*unused_args: Any, **unused_kwargs: Any) -> Any:
        raise AssertionError("Stability audit unexpectedly missed the warmed artifact cache")

    assert "_compile" not in vars(add_kernel)
    add_kernel._compile = unexpected_compile
    try:
        measure(probe, ablations, args)
    finally:
        try:
            probe.finish()
        finally:
            ablations.restore()
            del add_kernel._compile
            probe.state.close()


if __name__ == "__main__":
    main()
