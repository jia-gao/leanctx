"""Content-type classifier.

Given a chat message, returns a :class:`ContentType` label. The Router
uses that label to pick a Compressor: code, errors, and structured data
get verbatim, prose goes through LLMLingua-2, repeated tool outputs get
dropped.

v0.1 uses conservative heuristic rules — false positives route prose to
verbatim (costing compression opportunity) but false negatives risk
corrupting code. We err on the side of preservation.
"""

from __future__ import annotations

import hashlib
import re
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

# A JSON object key: a quoted token immediately followed by a colon. Used to
# measure how "object-shaped" a message is. Bounded length keeps a stray colon
# inside a long quoted prose sentence from registering as a key.
_JSON_KEY_RE = re.compile(r'"[^"\n]{1,64}"\s*:')

# Above this many JSON keys per 1000 characters, a message is treated as a
# serialized structured-data blob (session/dialogue logs, records, config)
# rather than prose. LLMLingua-2 is *extractive* — it drops low-information
# tokens — which shreds JSON structure (braces, keys, colons), corrupting the
# payload while reading as compressible prose. Such content routes to verbatim.
#
# Calibrated on LongBench v2: prose QA domains sit at ≤0.3 keys/1k chars while
# genuine JSON dialogue logs sit at ≥1.6; 1.0 falls cleanly in that gap. Tuned
# conservatively — a false positive only forgoes compression (safe), a false
# negative risks corruption (the failure this guards against).
_STRUCTURED_KEYS_PER_KCHAR = 1.0

# Below this length the key-density estimate is too noisy to trust: a short
# snippet with one or two key:value pairs is not a structured document.
_STRUCTURED_MIN_CHARS = 200


def classify(message: dict[str, Any]) -> ContentType:
    """Classify a single chat message by content shape.

    Check order: ERROR > CODE > STRUCTURED > PROSE. UNKNOWN when the
    message has no extractable text at all. STRUCTURED precedes PROSE so a
    JSON/structured-data blob is routed to verbatim rather than handed to
    the (extractive, structure-destroying) prose compressor.

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
    if _looks_like_structured(text):
        return ContentType.STRUCTURED
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


def _looks_like_structured(text: str) -> bool:
    """True when the message is a serialized structured-data blob (JSON).

    Measures JSON-key density rather than attempting a full parse: the payload
    is usually embedded in a prompt (document + question), so it never parses
    cleanly, but a high density of ``"key":`` tokens is an unambiguous shape
    signal that a lossy prose pass would corrupt.
    """
    n = len(text)
    if n < _STRUCTURED_MIN_CHARS:
        return False
    keys = len(_JSON_KEY_RE.findall(text))
    return keys / (n / 1000) >= _STRUCTURED_KEYS_PER_KCHAR


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
        # Skip messages that carry tool-use linkage: tool_use and
        # tool_result blocks pair by id, so dropping a "duplicate"
        # tool_result would orphan the matching tool_use in the
        # preceding assistant message. Always return "" here, which the
        # caller treats as "never flag this message".
        if _has_tool_linkage(message):
            return ""
        text = get_text_content(message)
        if not text:
            return ""
        # Include role so a user "ok" and an assistant "ok" don't collapse
        # into one message — those are distinct turns in the conversation.
        role = message.get("role", "")
        payload = f"{role}|{text}".encode()
        return hashlib.sha256(payload).hexdigest()


def _has_tool_linkage(message: dict[str, Any]) -> bool:
    """True if this message contains tool_use or tool_result blocks."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") in ("tool_use", "tool_result")
        for block in content
    )
