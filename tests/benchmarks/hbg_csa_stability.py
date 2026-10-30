# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Measure the real CSA fixture through this checkout's KERNEL eager entry.

The external fixture supplies formal layer weights and synthetic history. Its
Native numerical comparison remains separate from exact ablation equivalence.
No graph is captured. Each queue setting/seed must run in a fresh process.
"""

import argparse
import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any


class TileAblations:
    """Restore only the two annotation preparation functions between blocks."""

    def __init__(self, baseline: Path) -> None:
        from hbg_eager_latency import _DiagnosticPatches  # noqa: PLC0415

        self.patches = _DiagnosticPatches()
        self.specializer = importlib.import_module("pypto.jit.specializer")
        self.decorator = importlib.import_module("pypto.jit.decorator")
        names = ("_eager_tile_annotation_data", "tile_annotation_fingerprints")
        source = baseline.read_text()
        tree = ast.parse(source, filename=str(baseline))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert sorted(node.name for node in nodes) == sorted(names), (baseline, names)
        self.functions: dict[str, Any] = {}
        statements: list[ast.stmt] = list(nodes)
        # Function globals remain the live module, so the restored fingerprint
        # function sees the restored eager helper and current request snapshots.
        exec(
            compile(ast.Module(body=statements, type_ignores=[]), str(baseline), "exec"),
            vars(self.specializer),
            self.functions,
        )
        self.hashes = {str(baseline): hashlib.sha256(source.encode()).hexdigest()}
        for module in (self.specializer, self.decorator):
            assert module.__file__ is not None
            path = Path(module.__file__)
            self.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def install(self, stage: str) -> None:
        assert not self.patches.saved
        assert stage in ("current_a", "current_b", "off_tile", "off_tile_eager", "off_tile_strings"), stage
        if stage in ("off_tile", "off_tile_eager"):
            name = "_eager_tile_annotation_data"
            self.patches.set(self.specializer, name, self.functions[name])
        if stage in ("off_tile", "off_tile_strings"):
            name = "tile_annotation_fingerprints"
            self.patches.set(self.specializer, name, self.functions[name])
            # decorator imports this function by value rather than module lookup.
            self.patches.set(self.decorator, name, self.functions[name])

    def restore(self) -> None:
        self.patches.restore()


class FineAblations:
    """Separate existing groups without bypassing checks or request snapshots."""

    stages = (
        "current_a",
        "current_b",
        "off_constant_ast",
        "off_source_serialization",
        "off_variant_plan",
        "off_call_nodes",
        "off_signature",
        "off_tile_eager",
        "off_tile_strings",
        "off_session",
        "off_resolver",
        "off_initial_frame",
        "off_import",
        "off_source",
    )

    def __init__(self, baseline_root: Path, tile_baseline: Path) -> None:
        from hbg_stability_audit import Ablations  # noqa: PLC0415

        self.base = Ablations(baseline_root)
        self.tile = TileAblations(tile_baseline)
        # The plan builder and its caller changed their interface together.
        # Restore that pair together, keeping call-node analysis independent.
        groups = {
            "off_constant_ast": {"constant_dependency_names"},
            "off_source_serialization": {"_get_source_hash"},
            "off_variant_plan": {"_resolve_constexpr_bindings", "expand_constexpr_variants"},
            "off_call_nodes": {"_dep_call_nodes"},
        }
        for stage, names in groups.items():
            self.base.plans[stage] = [plan for plan in self.base.plans["off_structure"] if plan[1] in names]
            assert {plan[1] for plan in self.base.plans[stage]} == names, stage
        source = importlib.import_module("pypto.jit._source")
        self.base.modules[source.__name__] = source
        assert source.__file__ is not None
        path = Path(source.__file__)
        self.base.hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.base.plans["off_signature"] = []
        self.base.add("off_signature", "fc8248b^", source.__name__, "function_signature")
        self.hashes = self.base.hashes | self.tile.hashes

    def install(self, stage: str) -> None:
        assert stage in self.stages, stage
        if stage in ("off_tile_eager", "off_tile_strings"):
            self.tile.install(stage)
        else:
            self.base.install(stage)
            if stage == "off_signature":
                signature = self.base.modules["pypto.jit._source"].function_signature
                for name in ("pypto.jit._arguments", "pypto.jit.decorator"):
                    self.base.set(self.base.modules[name], "function_signature", signature)

    def restore(self) -> None:
        self.tile.restore()
        self.base.restore()


def bind_csa(source: Path, native: Path) -> Any:
    """Load only the CSA fixture dependencies, preserving installed PyPTO/runtime."""
    sys.path[:0] = [str(source), str(source / "tests/pypto_test"), str(native / "src/vllm")]

    class NativeExtensionFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname == "vllm_ascend.vllm_ascend_C":
                return importlib.machinery.PathFinder.find_spec(
                    fullname, [str(native / "src/vllm-ascend/vllm_ascend")]
                )
            return None

    sys.meta_path.insert(0, NativeExtensionFinder())
    environment: Any = importlib.import_module("dsv4_csa_env")
    environment.activate = lambda: source
    os.environ.update(DYNAMIC_EPLB="false", EXPERT_MAP_RECORD="false")
    case = importlib.import_module("dsv4_csa_single_layer")
    package = "vllm_ascend.ops.pypto.deepseek_v4_flash_csa"
    adapter = importlib.import_module(f"{package}.native_adapter")
    reduction = importlib.import_module(f"{package}.reduction")
    root = importlib.import_module(f"{package}.decode_csa").decode_csa_tp1_layer_hbg

    def register(cls: Any) -> Any:
        reduction.validate_reduction_mode()
        return cls(root)

    original_init = adapter.HBGNativeCSACall.__init__

    def initialize(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, kernel=root, **kwargs)

    adapter.HBGCSAOperators.register = classmethod(register)
    adapter.HBGNativeCSACall.__init__ = initialize
    return case, root


def check_state(
    case: Any, fixture: Any, output: Any, topk: Any, reference: Any, exact: bool
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    observed = case.collect_state(fixture, output, topk())
    guards = case.guard_checks(fixture)
    assert all(check["status"] == "PASS" for check in guards.values()), guards
    comparisons = {
        name: case.compare_tensor(value, reference[name], 0, 0) for name, value in observed.items()
    }
    assert all(check.get("nonfinite") == 0 for check in comparisons.values()), comparisons
    if exact:
        assert all(check["status"] == "PASS" for check in comparisons.values()), comparisons
    hashes = {
        name: hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name, value in observed.items()
    }
    return {"guards": guards, "comparison": comparisons, "hashes": hashes}


def measure_cpu(
    args: Any, stages: Any, ablations: Any, reset: Any, run: Any, drain: Any, verify: Any
) -> None:
    """Accumulate call/completion CPU and amortize one real release per group."""
    npu = importlib.import_module("torch_npu").npu
    rng = random.Random(args.seed + 1)
    lifecycle, orders = [], []
    for batch in range(args.blocks):
        order = list(stages)
        rng.shuffle(order)
        orders.append(order)
        for stage in order:
            drain()
            if ablations:
                ablations.restore()
                ablations.install(stage)
            cpu_ns = 0
            for _ in range(args.cpu_samples):
                reset()
                begin = time.process_time_ns()
                run()
                # Preserve the preceding call's leases until the next call;
                # synchronization completes device work, not source ownership.
                npu.synchronize()
                cpu_ns += time.process_time_ns() - begin
            begin = time.process_time_ns()
            drain()
            cpu_ns += time.process_time_ns() - begin
            checked = verify()
            lifecycle.append(
                {
                    "batch": batch,
                    "stage": stage,
                    "calls": args.cpu_samples,
                    "cpu_us": cpu_ns / args.cpu_samples / 1000,
                    "checks": checked,
                }
            )
            print("CSA_CPU_PASS", batch, stage, flush=True)
        (args.output / "pto-lifecycle.json").write_text(json.dumps(lifecycle))
        (args.output / "pto-lifecycle-orders.json").write_text(json.dumps(orders))


def diagnose_sources(
    args: Any, stages: Any, ablations: Any, root: Any, reset: Any, run: Any, drain: Any, verify: Any
) -> dict[str, Any]:
    """Count real source registration/release in a separate, untimed process."""
    from hbg_eager_latency import _DiagnosticPatches  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    state = get_process_kernel_state()
    session = state._worker
    rows = []
    for stage in stages:
        drain()
        if ablations:
            ablations.restore()
            ablations.install(stage)
        counters: dict[str, int] = {}
        patches = _DiagnosticPatches()

        def instrument(owner: Any, name: str, label: str) -> None:
            original = getattr(owner, name)
            counters[label] = 0

            def counted(*call_args: Any, **kwargs: Any) -> Any:
                counters[label] += 1
                return original(*call_args, **kwargs)

            patches.set(owner, name, counted)

        try:
            for owner, name, label in (
                (root, "_resolve_compiled", "resolver_calls"),
                (root, "_compile", "compile_attempts"),
                (session.sources, "acquire", "source_acquire"),
                (session.sources, "release", "source_release"),
                (session.sources, "_release_buffer", "source_buffer_release"),
                (session.worker, "_record_device_alloc", "consumer_register"),
                (session.worker, "_drop_device_alloc", "consumer_unregister"),
            ):
                instrument(owner, name, label)
            for _ in range(args.samples):
                reset()
                run()
            drain()
        finally:
            patches.restore()
        assert counters["resolver_calls"] == args.samples and counters["compile_attempts"] == 0, counters
        assert counters["consumer_register"] == counters["consumer_unregister"], counters
        rows.append({"stage": stage, "calls": args.samples, "counts": counters, "checks": verify()})
    result = {"method": "untimed_diagnostic", "us_p50": None, "counts": rows}
    (args.output / "source-counts.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    # The legacy fixture divides the returned p50 values. This internal sentinel
    # propagates "not measured"; final report loading converts it to JSON null.
    return result | {"us_p50": float("nan")}


def select_ablations(args: Any) -> tuple[Any, tuple[str, ...]]:
    from hbg_stability_audit import STAGES, Ablations  # noqa: PLC0415

    if args.mode == "smoke" or args.suite == "current":
        return None, ("current_a", "current_b")
    if args.suite == "tile":
        return TileAblations(args.tile_baseline), ("current_a", "current_b", "off_tile")
    if args.suite == "fine":
        return FineAblations(args.baselines, args.tile_baseline), FineAblations.stages
    return Ablations(args.baselines), tuple(stage for stage in STAGES if stage != "aclnn")


def install_measurement(case: Any, root: Any, args: Any) -> None:
    from hbg_stability_audit import npu  # noqa: PLC0415
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.runtime.kernel.arguments import _load_api  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    ablations, stages = select_ablations(args)
    rng = random.Random(args.seed)
    calls = 0
    cold_compile = []
    original_compile = root._compile

    def time_compile(*call_args: Any, **kwargs: Any) -> Any:
        begin = time.perf_counter_ns()
        try:
            return original_compile(*call_args, **kwargs)
        finally:
            cold_compile.append({"wall_ms": (time.perf_counter_ns() - begin) / 1e6})
            (args.output / "cold-compile.json").write_text(json.dumps(cold_compile))

    root._compile = time_compile
    if ablations:
        (args.output / "ablation-source-hashes.json").write_text(json.dumps(ablations.hashes, indent=2))

    def measure(
        fixture: Any,
        run: Any,
        output: Any,
        topk: Any,
        reference: Any,
        *,
        iters: int,
        warmup: int,
        require_exact: bool,
        profile_dir: Any = None,
    ) -> dict[str, Any]:
        nonlocal calls
        assert (
            logging.getLogger("simpler").getEffectiveLevel()
            == importlib.import_module("simpler._log").get_current_config()
            == logging.ERROR
        )
        calls += 1
        assert calls in (1, 2), f"Unexpected timing callback {calls}"
        is_pto = calls == 2
        assert profile_dir is None, "Formal timing must not enable profiling"
        initial = {
            name: group["initial"].to(group["allocation"].device) for name, group in fixture["groups"].items()
        }
        state = get_process_kernel_state()
        artifact_ids = tuple(id(value) for value in root._cache.values())
        if is_pto:
            assert len(artifact_ids) == 1, f"Expected one compiled CSA artifact, got {artifact_ids}"
            for artifact in root._cache.values():
                artifact.execution_capabilities.require(ArtifactExecutionMode.KERNEL)

            def reject_compile(*unused_args: Any, **unused_kwargs: Any) -> Any:
                raise AssertionError("CSA recompiled during a fixed-artifact measurement")

            assert root._compile is time_compile
            root._compile = reject_compile

        def reset() -> None:
            for name, group in fixture["groups"].items():
                group["allocation"].copy_(initial[name])
            output.fill_(float("nan"))
            npu.synchronize()

        def drain() -> None:
            npu.synchronize()
            if is_pto:
                state.drain()
                session = state._worker
                worker = session.worker
                # Complete lazy registry revocations as well as source leases.
                with worker._child_prov_worker_lock(0), worker._child_prov_lock:
                    assert worker._attach_source_buffers_locked(_load_api().TaskArgs()) == ()
                    assert not worker._source_attachments
                    assert not worker._device_buffer_source._records
                assert not session.sources._sources

        expected_hashes = None
        rows, orders, checks = [], [], []
        selected = stages if is_pto else ("native",)

        def verify() -> dict[str, Any]:
            checked = check_state(case, fixture, output, topk, reference, require_exact)
            if expected_hashes is not None:
                assert checked["hashes"] == expected_hashes, "CSA lifecycle state differs from Host pass"
            return checked

        try:
            if is_pto and args.mode == "diagnostic":
                return diagnose_sources(args, stages, ablations, root, reset, run, drain, verify)
            for batch in range(args.blocks if is_pto else 1):
                order = list(selected)
                rng.shuffle(order)
                orders.append(order)
                for stage in order:
                    drain()
                    if is_pto and ablations:
                        ablations.restore()
                        ablations.install(stage)
                    for _ in range(warmup):
                        reset()
                        run()
                        npu.synchronize()
                    for sample in range(iters):
                        reset()
                        begin = time.perf_counter_ns()
                        run()
                        end = time.perf_counter_ns()
                        npu.synchronize()
                        rows.append(
                            {
                                "batch": batch,
                                "stage": stage,
                                "sample": sample,
                                "host_us": (end - begin) / 1000,
                            }
                        )
                    drain()
                    checked = check_state(case, fixture, output, topk, reference, require_exact)
                    if expected_hashes is not None:
                        assert checked["hashes"] == expected_hashes, "CSA state differs across blocks or arms"
                    expected_hashes = checked["hashes"]
                    assert not is_pto or tuple(id(value) for value in root._cache.values()) == artifact_ids
                    checks.append({"batch": batch, "stage": stage, **checked})
                    print("CSA_HOST_PASS", batch, stage, flush=True)
                    (args.output / ("pto-host.json" if is_pto else "native-host.json")).write_text(
                        json.dumps(rows)
                    )
            if is_pto:
                measure_cpu(args, stages, ablations, reset, run, drain, verify)
        finally:
            try:
                drain()
            finally:
                if is_pto:
                    del root._compile
                    if ablations:
                        ablations.restore()
        result = {
            "method": "eager_host_entry_to_return",
            "scope": (
                "Reset/synchronization excluded from Host; lifecycle includes completion and source release"
            ),
            "orders": orders,
            "checks": checks,
            "samples_us": [row["host_us"] for row in rows],
            "us_p50": statistics.median(
                row["host_us"] for row in rows if row["stage"] in ("current_a", "current_b", "native")
            ),
        }
        (args.output / ("pto-checks.json" if is_pto else "native-checks.json")).write_text(
            json.dumps(result, indent=2)
        )
        return result

    case.measure_graph_interval = measure


def write_identity(args: Any) -> None:
    modules = {}
    for name in (
        "pypto",
        "pypto.pypto_core",
        "pypto._torch_npu",
        "simpler",
        "_task_interface",
        "torch",
        "torch_npu",
        "vllm",
        "vllm_ascend",
    ):
        module = importlib.import_module(name)
        assert module.__file__ is not None, f"Module {name} has no source path"
        path = Path(module.__file__).resolve()
        modules[name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    repo = Path(__file__).resolve().parents[2]
    assert Path(modules["pypto"]["file"]).is_relative_to(repo)
    assert Path(modules["simpler"]["file"]).is_relative_to(repo / "runtime")
    identity = {
        "modules": modules,
        "pid": os.getpid(),
        "command_line": sys.argv,
        "simpler_log_level": logging.getLogger("simpler").getEffectiveLevel(),
        "measurement": {
            name: getattr(args, name)
            for name in ("mode", "suite", "seed", "blocks", "samples", "warmup", "cpu_samples", "history")
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TASK_QUEUE_ENABLE",
                "PTOAS_ROOT",
                "ASCEND_HOME_PATH",
                "PYTHONNOUSERSITE",
                "PYPTO_CACHE",
                "ASCEND_RT_VISIBLE_DEVICES",
            )
        },
    }
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--suite", choices=("all", "tile", "fine", "current"), default="current")
    parser.add_argument("--tile-baseline", type=Path, help="Pre-optimization specializer.py for tile/fine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "bench", "diagnostic"), default="smoke")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--history", type=int, default=8192)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--cpu-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.mode != "smoke" and args.suite in ("all", "fine"):
        parser.error("all/fine include removed optimizations; run those historical suites on the bak branch")
    if args.mode != "smoke":
        if args.suite in ("all", "fine") and args.baselines is None:
            parser.error("--suite all/fine with --mode bench/diagnostic requires --baselines")
        if args.suite in ("tile", "fine") and args.tile_baseline is None:
            parser.error("--suite tile/fine with --mode bench/diagnostic requires --tile-baseline")
    if min(args.blocks, args.samples, args.warmup, args.cpu_samples) < 1:
        parser.error("All measurement counts must be positive")
    if args.mode == "smoke":
        args.blocks, args.samples, args.warmup, args.cpu_samples = 1, 2, 1, 1
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["PYPTO_CACHE"] = "0"
    os.environ.update(
        PTO_CSA_VARIANT="performance",
        VLLM_ASCEND_ENABLE_NZ="2",
        VLLM_ASCEND_PTO_CSA_ATOMIC_ADD="0",
        HCCL_DETERMINISTIC="true",
    )
    case, root = bind_csa(args.source.resolve(), args.native.resolve())
    install_measurement(case, root, args)
    # Worker.init snapshots this threshold into native logging. Configure it
    # before the fixture creates the Worker, not merely before timed calls.
    importlib.import_module("simpler_setup.log_config").configure_logging("error")
    write_identity(args)
    sys.argv = [
        str(args.source / "tests/pypto_test/dsv4_csa_single_layer.py"),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--device",
        str(args.device),
        "--layer-index",
        "2",
        "--batch",
        "4",
        "--history",
        str(args.history),
        "--runtime",
        "host_build_graph",
        "--variant",
        "performance",
        "--weight-nz-mode",
        "2",
        "--atomic-add",
        "0",
        "--deterministic-level",
        "1",
        "--timing-iters",
        str(args.samples),
        "--timing-warmup",
        str(args.warmup),
        "--timing-metadata",
        "reuse",
    ]
    try:
        case.main()
    finally:
        from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

        get_process_kernel_state().close()
    path = args.output / "report.json"
    report = json.loads(path.read_text(), parse_constant=lambda value: None)
    report["execution_config"]["path"] = "kernel_eager"
    if args.mode == "diagnostic":
        report["timing"].pop("native_over_pto_p50", None)
        report["timing"]["pto"].pop("us_p50", None)
    report["timing"].update(scope="CSA KERNEL eager Host timing", method="No graph capture or replay")
    path.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
