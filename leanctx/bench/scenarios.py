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


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    # Import runner modules so their @register decorators fire. Each
    # runner module is responsible for its own optional-import
    # handling — the registry call should NOT import optional extras.
    from leanctx.bench.runners import (  # noqa: F401
        agent_structural,
        anthropic_e2e,
        lingua_local,
    )


def reset_for_tests() -> None:
    """Clear the registry — used by tests that register custom scenarios."""
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
