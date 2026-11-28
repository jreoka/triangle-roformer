"""torch.compile + activation-checkpointing helpers.

Calling `checkpoint(compiled_module, ...)` re-enters Dynamo during backward to
recompute the forward. On CUDA that recompute runs on autograd's device thread,
whose global state (e.g. `torch.get_num_threads()`) differs from the forward
thread, so Dynamo recompiles every graph a second time. Once the recompile limit
is reached it falls back to eager for the recompute, whose saved tensors no
longer match the compiled forward (`CheckpointError`).

`compile_unit` instead compiles `checkpoint(module.forward, ...)` as one region:
Dynamo captures the checkpoint as a higher-order op and the recompute becomes
part of the compiled backward graph, so backward never re-enters Dynamo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

_COMPILED_CKPT_ATTR = "_triangle_compiled_checkpoint"


def configure_dynamo_cache(limit: int = 64, accumulated_limit: int = 256) -> None:
    """Raise Dynamo's per-function and accumulated recompile limits (both the
    current `recompile_limit` names and the legacy `cache_size_limit` names)."""
    if not hasattr(torch, "_dynamo"):
        return
    import torch._dynamo.config as dynamo_config

    for names, value in (
        (("recompile_limit", "cache_size_limit"), limit),
        (("accumulated_recompile_limit", "accumulated_cache_size_limit"), accumulated_limit),
    ):
        for name in names:
            if hasattr(dynamo_config, name):
                current = int(getattr(dynamo_config, name))
                setattr(dynamo_config, name, max(current, value))


def compile_unit(
    module: nn.Module,
    mode: str = "default",
    dynamic: bool | None = False,
) -> None:
    """Compile `module` for plain calls and, separately, a checkpointed call
    (checkpoint inside the compiled region) used by `run_unit` when
    activation checkpointing is on."""
    module.compile(mode=mode, dynamic=dynamic)
    forward = module.forward

    def checkpointed_call(*args: Any) -> Any:
        return checkpoint(forward, *args, use_reentrant=False)

    # Plain attribute (not a submodule): stays out of state_dict / named_modules.
    object.__setattr__(
        module,
        _COMPILED_CKPT_ATTR,
        torch.compile(checkpointed_call, mode=mode, dynamic=dynamic),
    )


def run_unit(module: nn.Module, use_checkpoint: bool, *args: Any) -> Any:
    """Call `module(*args)`, optionally with activation checkpointing; uses the
    compiled checkpointed call when `compile_unit` was applied."""
    if not use_checkpoint:
        return module(*args)
    compiled = module.__dict__.get(_COMPILED_CKPT_ATTR)
    if compiled is not None:
        return compiled(*args)
    return checkpoint(module, *args, use_reentrant=False)
