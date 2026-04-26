"""Span-owning iterator wrappers for streaming wrapper paths.

For stream-returning wrapper paths (OpenAI ``stream=True``, Gemini
``generate_content_stream``, Gemini async stream), the leanctx wrapper
must close the span at the FIRST of:

- iterator exhaustion (StopIteration / StopAsyncIteration),
- explicit ``.close()`` on the wrapper (forwarded to the upstream),
- ``__del__`` finalization (GC backstop, for the abandoned-iterator case).

These wrappers take ownership of an already-detached
``_LeanctxSpan`` (see ``compression_span.detach_span``) and the
upstream iterator. They invoke ``span.close()`` exactly once, no
matter which close path fires first.

Iteration-time exceptions are recorded on the span via
``set_error`` before the span is closed, then re-raised.
"""

from __future__ import annotations

from typing import Any


class _SpanOwningIterator:
    """Wraps a sync iterator/Stream so the span closes when iteration ends."""

    def __init__(self, upstream: Any, span: Any) -> None:
        self._upstream = upstream
        self._span = span
        self._closed = False

    def __iter__(self) -> _SpanOwningIterator:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._upstream)
        except StopIteration:
            self.close()
            raise
        except BaseException as exc:
            if not self._closed:
                self._span.set_error(exc)
                self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        upstream_close = getattr(self._upstream, "close", None)
        try:
            if callable(upstream_close):
                upstream_close()
        finally:
            self._span.close()

    def __enter__(self) -> _SpanOwningIterator:
        upstream_enter = getattr(self._upstream, "__enter__", None)
        if callable(upstream_enter):
            upstream_enter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        upstream_exit = getattr(self._upstream, "__exit__", None)
        try:
            if callable(upstream_exit):
                upstream_exit(exc_type, exc_val, exc_tb)
        finally:
            if not self._closed:
                if exc_val is not None:
                    self._span.set_error(exc_val)
                self.close()

    def __del__(self) -> None:
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)


class _SpanOwningAsyncIterator:
    """Async variant of :class:`_SpanOwningIterator`."""

    def __init__(self, upstream: Any, span: Any) -> None:
        self._upstream = upstream
        self._span = span
        self._closed = False

    def __aiter__(self) -> _SpanOwningAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._upstream.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException as exc:
            if not self._closed:
                self._span.set_error(exc)
                await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        upstream_aclose = getattr(self._upstream, "aclose", None)
        upstream_close = getattr(self._upstream, "close", None)
        try:
            if callable(upstream_aclose):
                await upstream_aclose()
            elif callable(upstream_close):
                upstream_close()
        finally:
            self._span.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        upstream_close = getattr(self._upstream, "close", None)
        try:
            if callable(upstream_close):
                upstream_close()
        finally:
            self._span.close()

    async def __aenter__(self) -> _SpanOwningAsyncIterator:
        upstream_aenter = getattr(self._upstream, "__aenter__", None)
        if callable(upstream_aenter):
            await upstream_aenter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        upstream_aexit = getattr(self._upstream, "__aexit__", None)
        try:
            if callable(upstream_aexit):
                await upstream_aexit(exc_type, exc_val, exc_tb)
        finally:
            if not self._closed:
                if exc_val is not None:
                    self._span.set_error(exc_val)
                await self.aclose()

    def __del__(self) -> None:
        try:
            if not self._closed:
                self.close()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upstream, name)
