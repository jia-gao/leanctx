"""Router — picks a Compressor for a given ContentType.

Used by the Middleware pipeline after classification:

    for msg in messages:
        ctype = classify(msg)
        c = router.route(ctype)
        out, stats = c.compress([msg])

Safe default: any unmapped ContentType falls back to :class:`Verbatim`.
Never corrupts content just because a configuration entry is missing.
"""

from __future__ import annotations

from leanctx.compressors import Compressor, ContentType, Verbatim


class Router:
    """Static mapping from :class:`ContentType` to :class:`Compressor`."""

    def __init__(
        self,
        routes: dict[ContentType, Compressor] | None = None,
        default: Compressor | None = None,
    ) -> None:
        self._routes: dict[ContentType, Compressor] = dict(routes or {})
        self._default: Compressor = default if default is not None else Verbatim()

    @property
    def default(self) -> Compressor:
        return self._default

    def register(self, content_type: ContentType, compressor: Compressor) -> None:
        """Register (or overwrite) a mapping after construction."""
        self._routes[content_type] = compressor

    def route(self, content_type: ContentType) -> Compressor:
        """Return the Compressor for a ContentType, or the default."""
        return self._routes.get(content_type, self._default)
