"""AC-3 (full taxonomy via spans), AC-5 (e2e cost), AC-6 (parentage)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from leanctx.client import _Messages
from leanctx.compressors import Verbatim
from leanctx.middleware import Middleware
from leanctx.observability.config import ObservabilityConfig
from leanctx.stats import CompressionStats
from tests.observability.conftest import otel_required


def _obs() -> ObservabilityConfig:
    return ObservabilityConfig(otel=True)


class _FakeSelfLLM:
    """Stand-in compressor that returns SelfLLM-shaped stats with a cost.

    Used to verify AC-5 e2e cost flow without hitting a real provider.
    """

    name = "selfllm"

    def __init__(self, *, cost_usd: float = 0.0123) -> None:
        self._cost = cost_usd

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats(
            input_tokens=100,
            output_tokens=30,
            ratio=0.3,
            method="selfllm",
            cost_usd=self._cost,
        )

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)


def _mock_response() -> Any:
    class _Usage:
        input_tokens = 0
        output_tokens = 0

    class _R:
        pass

    r = _R()
    r.usage = _Usage()  # type: ignore[attr-defined]
    return r


@otel_required
def test_taxonomy_below_threshold_observable(spans: Any) -> None:
    """AC-3: mode=on but message tokens < threshold → method=below-threshold."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 10_000_000}},
        observability=obs,
    )
    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(messages=[{"role": "user", "content": "small"}])

    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "leanctx.compress")
    assert root.attributes["leanctx.method"] == "below-threshold"


@otel_required
def test_taxonomy_verbatim_observable(spans: Any) -> None:
    """AC-3: pipeline ran and only Verbatim used → method=verbatim."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}, "routing": {}},
        observability=obs,
    )
    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(
        messages=[
            {"role": "user", "content": "a long enough message to exceed threshold"}
        ]
    )

    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "leanctx.compress")
    assert root.attributes["leanctx.method"] == "verbatim"


@otel_required
def test_taxonomy_selfllm_observable_via_router(spans: Any) -> None:
    """AC-3: pipeline routes to a SelfLLM-shaped compressor → method=selfllm."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )
    # Inject a fake SelfLLM via the router so we don't need real API.
    from leanctx.compressors.base import ContentType

    mw._router.register(ContentType.PROSE, _FakeSelfLLM())  # type: ignore[attr-defined]

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(messages=[{"role": "user", "content": "prose-like content here"}])

    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "leanctx.compress")
    assert root.attributes["leanctx.method"] == "selfllm"
    assert root.attributes["leanctx.cost_usd"] == 0.0123


@otel_required
def test_taxonomy_hybrid_observable_via_two_compressors(spans: Any) -> None:
    """AC-3: two different compressors used → method=hybrid (set union > 1)."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )
    from leanctx.compressors.base import ContentType

    # Route prose to fake-selfllm; default (UNKNOWN/CODE) stays Verbatim.
    mw._router.register(ContentType.PROSE, _FakeSelfLLM(cost_usd=0.05))  # type: ignore[attr-defined]
    mw._router.register(ContentType.CODE, Verbatim())  # type: ignore[attr-defined]

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(
        messages=[
            {"role": "user", "content": "some prose content here"},
            {"role": "user", "content": "```py\ndef f(): pass\n```"},
        ]
    )

    finished = spans.get_finished_spans()
    root = next(s for s in finished if s.name == "leanctx.compress")
    assert root.attributes["leanctx.method"] == "hybrid"
    # Hybrid cost is the sum of constituents (selfllm contributed 0.05).
    assert root.attributes["leanctx.cost_usd"] == 0.05


@otel_required
def test_e2e_cost_single_increment_for_nested_call(spans: Any) -> None:
    """AC-5 e2e: wrapper-routed selfllm call increments cost counter exactly
    ONCE — not once at wrapper, once at middleware, once at compressor."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )
    from leanctx.compressors.base import ContentType

    mw._router.register(ContentType.PROSE, _FakeSelfLLM(cost_usd=0.0042))  # type: ignore[attr-defined]
    mw._router.register(ContentType.UNKNOWN, _FakeSelfLLM(cost_usd=0.0042))  # type: ignore[attr-defined]

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(messages=[{"role": "user", "content": "prose-shaped content"}])

    finished = spans.get_finished_spans()
    # Exactly ONE leanctx.compress span (wrapper) with cost.
    compress_spans = [s for s in finished if s.name == "leanctx.compress"]
    assert len(compress_spans) == 1
    assert compress_spans[0].attributes["leanctx.cost_usd"] == 0.0042

    # Middleware did NOT emit a duplicate compress span (depth-counter
    # suppression). Wrapper-routed = exactly one root.


@otel_required
def test_parentage_compressor_child_of_wrapper_root(spans: Any) -> None:
    """AC-6: leanctx.compressor.compress spans are children of the
    leanctx.compress root from the wrapper, with parent_span_id linkage."""
    obs = _obs()
    mw = Middleware(
        {"mode": "on", "trigger": {"threshold_tokens": 1}},
        observability=obs,
    )

    upstream = MagicMock()
    upstream.messages.create = MagicMock(return_value=_mock_response())
    wrapper = _Messages(upstream, mw, obs)

    wrapper.create(messages=[{"role": "user", "content": "prose-shaped content"}])

    finished = spans.get_finished_spans()
    roots = [s for s in finished if s.name == "leanctx.compress"]
    children = [s for s in finished if s.name == "leanctx.compressor.compress"]
    assert len(roots) == 1
    assert len(children) >= 1
    root_span_id = roots[0].context.span_id
    for child in children:
        assert child.parent is not None, (
            f"child compressor span {child.attributes} is orphaned"
        )
        assert child.parent.span_id == root_span_id
