"""Router tests — static routing and default fallback."""

from typing import Any

from leanctx import CompressionStats, ContentType, Router, Verbatim


class _StubCompressor:
    """Minimal Compressor used to verify routing picks the right instance."""

    def __init__(self, label: str) -> None:
        self.name = label

    def compress(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return messages, CompressionStats(method=self.name)

    async def compress_async(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], CompressionStats]:
        return self.compress(messages)


def test_empty_router_uses_verbatim_default() -> None:
    r = Router()
    assert isinstance(r.default, Verbatim)
    assert r.route(ContentType.PROSE).name == "verbatim"


def test_custom_default_is_honored() -> None:
    sentinel = _StubCompressor("custom-default")
    r = Router(default=sentinel)
    assert r.default is sentinel
    assert r.route(ContentType.UNKNOWN) is sentinel


def test_constructor_routes_are_applied() -> None:
    prose = _StubCompressor("prose")
    code = _StubCompressor("code")
    r = Router({ContentType.PROSE: prose, ContentType.CODE: code})
    assert r.route(ContentType.PROSE) is prose
    assert r.route(ContentType.CODE) is code


def test_unmapped_type_falls_back_to_default() -> None:
    prose = _StubCompressor("prose")
    r = Router({ContentType.PROSE: prose})
    # ERROR has no mapping → default (Verbatim).
    assert r.route(ContentType.ERROR).name == "verbatim"


def test_register_adds_mapping() -> None:
    r = Router()
    lingua_stub = _StubCompressor("lingua")
    r.register(ContentType.PROSE, lingua_stub)
    assert r.route(ContentType.PROSE) is lingua_stub


def test_register_overwrites_prior_mapping() -> None:
    first = _StubCompressor("first")
    second = _StubCompressor("second")
    r = Router({ContentType.PROSE: first})
    r.register(ContentType.PROSE, second)
    assert r.route(ContentType.PROSE) is second


def test_route_output_drives_compression() -> None:
    # End-to-end sanity: Router picks a Compressor, that Compressor's
    # compress() returns stats whose method matches the chosen compressor.
    r = Router({ContentType.PROSE: _StubCompressor("prose-stub")})
    compressor = r.route(ContentType.PROSE)
    _, stats = compressor.compress([{"role": "user", "content": "hi"}])
    assert stats.method == "prose-stub"
