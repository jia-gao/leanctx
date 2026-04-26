"""Scenario registry for bench runners.

Each scenario is registered via the ``@register`` decorator. Importing
this module triggers registration of the runners shipped with leanctx;
out-of-tree consumers can register their own scenarios by importing
``register`` and decorating their runner function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from leanctx.bench.schema import BenchRecord


@dataclass(frozen=True)
class ScenarioInfo:
    name: str
    description: str
    required_extras: tuple[str, ...] = ()
    required_env: tuple[str, ...] = ()


_REGISTRY: dict[str, tuple[ScenarioInfo, Callable[..., BenchRecord]]] = {}


def register(
    name: str,
    *,
    description: str = "",
    required_extras: tuple[str, ...] = (),
    required_env: tuple[str, ...] = (),
) -> Callable[[Callable[..., BenchRecord]], Callable[..., BenchRecord]]:
    """Decorator: register a runner function under a scenario name."""

    def deco(fn: Callable[..., BenchRecord]) -> Callable[..., BenchRecord]:
        _REGISTRY[name] = (
            ScenarioInfo(
                name=name,
                description=description,
                required_extras=required_extras,
                required_env=required_env,
            ),
            fn,
        )
        return fn

    return deco


def list_scenarios() -> list[ScenarioInfo]:
    _ensure_loaded()
    return [info for info, _ in _REGISTRY.values()]


def get(name: str) -> tuple[ScenarioInfo, Callable[..., BenchRecord]]:
    _ensure_loaded()
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown scenario {name!r}; known: {known}")
    return _REGISTRY[name]


_LOADED = False

# Import names of the runner modules whose @register decorators populate
# the built-in registry. Listed here (rather than written as imports)
# so reset_for_tests() can re-trigger registration via importlib.reload.
_BUILTIN_RUNNER_MODULES = (
    "leanctx.bench.runners.agent_structural",
    "leanctx.bench.runners.anthropic_e2e",
    "leanctx.bench.runners.lingua_local",
    "leanctx.bench.runners.selfllm_provider",
)


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    for modname in _BUILTIN_RUNNER_MODULES:
        if modname in sys.modules:
            # Already imported in this process — reload to re-fire
            # @register decorators after a reset_for_tests() cycle.
            importlib.reload(sys.modules[modname])
        else:
            importlib.import_module(modname)


def reset_for_tests() -> None:
    """Clear the registry — used by tests that register custom scenarios.

    After reset, the next call to a public function (`list_scenarios`,
    `get`) will re-import the built-in runners so the canonical six
    scenarios reappear. Tests that need a clean slate WITHOUT the
    built-ins can register their own scenarios immediately after the
    reset (the built-ins won't be loaded until a public accessor is
    called).
    """
    global _LOADED
    _REGISTRY.clear()
    _LOADED = False


def _register_for_tests(
    name: str,
    info: ScenarioInfo,
    fn: Callable[..., BenchRecord],
) -> None:
    _REGISTRY[name] = (info, fn)


def _build_runtime_info(name: str) -> dict[str, Any]:
    """Return diagnostic info about a scenario's runtime requirements."""
    _ensure_loaded()
    info, _ = _REGISTRY[name]
    return {
        "name": info.name,
        "description": info.description,
        "required_extras": list(info.required_extras),
        "required_env": list(info.required_env),
    }
