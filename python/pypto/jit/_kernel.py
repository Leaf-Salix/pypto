# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Eager JIT policy over the shared specialization, compiler and artifact cache."""

from functools import lru_cache
from typing import Any

from pypto._artifact_contract import ArtifactExecutionMode
from pypto.runtime.kernel.context import get_process_kernel_state


@lru_cache(maxsize=128)
def _kernel_signature(parameters: tuple[Any, ...], aliases: tuple[int, ...], platform: str) -> Any:
    """Reuse only immutable parameter declarations, never a call's frame or owners.

    Metadata snapshots include shapes, dtypes, directions and layouts. Aliases
    and platform are also part of the key. Bound the cache so discarded artifact
    signatures cannot accumulate indefinitely; eviction only repeats parsing.
    """
    from pypto.torch.interop import CallSignature  # noqa: PLC0415

    return CallSignature(parameters, return_aliases=aliases, platform=platform)


def invoke_jit(jit: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    # This gate intentionally precedes binding, dependency discovery and imports
    # of optional Torch/native execution dependencies.
    state = get_process_kernel_state()
    bound = state.require_config()
    from pypto.ir._compile_contract import analyze_contract  # noqa: PLC0415
    from pypto.runtime import CompileOptions  # noqa: PLC0415
    from pypto.torch import launch  # noqa: PLC0415
    from pypto.torch.interop import CallSignature  # noqa: PLC0415

    options = kwargs.get("config")
    if options is None:
        options = CompileOptions()
    if not isinstance(options, CompileOptions):
        raise TypeError("Direct JIT kernel calls require CompileOptions; use op.compile() for RunConfig")
    if options.distributed_config is not None:
        raise ValueError("Direct JIT kernel calls do not support CompileOptions.distributed_config")
    capturing = launch.check_eager_call(bound)

    def preflight(program: Any, ordered: list[Any]) -> None:
        # On a miss, validate the parsed logical signature before passes/codegen.
        # On a hit the same validation reads the artifact's persisted metadata.
        params, aliases = analyze_contract(program, ArtifactExecutionMode.KERNEL)
        assert params is not None and aliases is not None
        frame = CallSignature(params, return_aliases=aliases, platform=bound.platform).describe_call(ordered)
        launch.check_frame(frame, bound)

    artifact, ordered, _ = jit._resolve_compiled(
        args,
        kwargs | {"config": options},
        _contract=ArtifactExecutionMode.KERNEL,
        _preflight=preflight,
        _lookup_only=capturing,
        _kernel_config=bound,
    )
    artifact.execution_capabilities.require(ArtifactExecutionMode.KERNEL)
    metadata = artifact.metadata
    if metadata is None:
        raise ValueError("Kernel artifact has no validated parameter metadata; recompile it")
    assert metadata.return_aliases is not None
    frame = _kernel_signature(metadata.parameters, metadata.return_aliases, bound.platform).describe_call(
        ordered
    )
    # invoke validates this exact frame before registration/admission. Keep the
    # early capture gate and cold compiler preflight, but do not validate the
    # final frame twice on every cache hit.
    return launch.invoke(artifact, frame, bound)
