"""Content-type classifier.

Given a chat message, returns a :class:`ContentType` label. The Router
uses that label to pick a Compressor: code and errors get verbatim,
prose goes through LLMLingua-2, repeated tool outputs get dropped.

v0.1 uses conservative heuristic rules — false positives route prose to
verbatim (costing compression opportunity) but false negatives risk
corrupting code. We err on the side of preservation.
"""

from __future__ import annotations

import hashlib
from typing import Any

from leanctx._content import get_text_content
from leanctx.compressors import ContentType

# Phrases that indicate an error, traceback, or panic. Ordered roughly by
# how strong a signal each is — the first match wins.
_ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "Uncaught exception",
    "UnhandledPromiseRejection",
    "thread 'main' panicked at",
    "panic: ",
    "Exception in thread",
    "FATAL: ",
    "ERROR: ",
    "Error: ",
    "Exception: ",
    "java.lang.",
)

# Tokens that strongly suggest source code at a line start.
_CODE_LINE_PREFIXES = (
    "def ",
    "class ",
    "function ",
    "async function ",
    "import ",
    "from ",
    "export ",
    "package ",
    "#include",
    "fn ",
    "pub fn ",
    "func ",
    "var ",
    "const ",
    "let ",
)

# Minimum code-like lines before a message is classified as CODE (when no
# fenced block is present). Two is enough to avoid single-keyword prose
# (e.g. "the import statement is tricky") from flipping the classification.
_MIN_CODE_LINES = 2


def classify(message: dict[str, Any]) -> ContentType:
    """Classify a single chat message by content shape.

    Check order: ERROR > CODE > PROSE. UNKNOWN when the message has no
    extractable text at all.

    Repeat detection is stateful and lives in :class:`RepeatTracker` —
    callers combine that with :func:`classify` to decide routing.
    """
    text = get_text_content(message)
    if not text.strip():
        return ContentType.UNKNOWN
    if _looks_like_error(text):
        return ContentType.ERROR
    if _looks_like_code(text):
        return ContentType.CODE
    return ContentType.PROSE


def _looks_like_error(text: str) -> bool:
    return any(marker in text for marker in _ERROR_MARKERS)


def _looks_like_code(text: str) -> bool:
    # Fenced code blocks are an unambiguous signal.
    if "```" in text:
        return True
    lines = text.splitlines()
    code_lines = sum(
        1
        for line in lines
        if any(line.lstrip().startswith(prefix) for prefix in _CODE_LINE_PREFIXES)
    )
    return code_lines >= _MIN_CODE_LINES


class RepeatTracker:
    """Tracks content hashes across a session to flag duplicate messages.

    The same tool call with the same arguments often produces the same
    output across turns (think repeated ``grep`` queries, ``ls``, status
    checks). Flagging duplicates lets the Router drop all but the most
    recent copy.

    Not safe for concurrent use — instantiate one per session.
    """

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_repeat(self, message: dict[str, Any]) -> bool:
        """Return ``True`` if this message's content has been seen before.

        Also records the content as seen, so a subsequent identical
        message still reports True.
        """
        h = self._hash(message)
        if h == "":
            return False
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def reset(self) -> None:
        self._seen.clear()

    @staticmethod
    def _hash(message: dict[str, Any]) -> str:
        text = get_text_content(message)
        if not text:
            return ""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
