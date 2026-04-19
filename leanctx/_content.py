"""Shared helpers for reading chat-message content.

OpenAI-style messages carry a plain string in ``content``. Anthropic-style
messages may carry a list of typed blocks — ``text``, ``image``,
``tool_use``, ``tool_result``, ``document``, ``thinking``. Both providers
use nested blocks for tool-use flows: a ``tool_result`` block's
``content`` can itself be a string or another list of blocks.

Token counting, the classifier, dedup hashing, and purge logic all need
to see the full compressible text — not just top-level text blocks.
This module centralizes the recursive extraction.

Non-text content types (``image``, ``thinking``, control blocks) return
empty strings. ``tool_use`` blocks serialize their ``input`` dict so tool
arguments (often large — full file contents, long search queries) are
visible to the compression trigger.
"""

from __future__ import annotations

import json
from typing import Any


def get_text_content(message: dict[str, Any]) -> str:
    """Return the concatenated compressible text from a chat message.

    Handles:
    * string content (OpenAI-style)
    * list of blocks (Anthropic-style), including:
        - ``text`` — included verbatim
        - ``tool_use`` — input serialized as JSON, prefixed with tool name
        - ``tool_result`` — nested content recursively extracted
        - ``document`` — either ``text`` field or ``source.data``, or
          nested ``content``
        - ``image``, ``thinking``, unknown types — contribute nothing

    Blocks that look malformed (missing ``type``, non-dict) silently
    contribute empty strings rather than raising — the caller is usually
    a trigger/classifier pipeline that should degrade gracefully.
    """
    return _extract_text(message.get("content"))


def _extract_text(content: Any) -> str:
    """Recursively turn any content shape into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            t = _text_from_block(item)
            if t:
                parts.append(t)
        return "\n".join(parts)
    if isinstance(content, dict):
        # Some providers wrap a single block in an object rather than a list.
        return _text_from_block(content)
    return ""


def _text_from_block(block: Any) -> str:
    """Extract text from a single structured content block."""
    if not isinstance(block, dict):
        return ""
    btype = block.get("type", "")

    if btype == "text":
        text = block.get("text", "")
        return text if isinstance(text, str) else ""

    if btype == "tool_use":
        # Tool call arguments. These can be large (full file paths, long
        # search queries, whole diffs as input) and should count toward
        # the compression trigger. Prefix with the tool name so the
        # compressor's view matches the agent's mental model.
        name = block.get("name", "")
        serialized = _serialize(block.get("input"))
        if name and serialized:
            return f"[tool_use {name}] {serialized}"
        return serialized

    if btype == "tool_result":
        # The tool's output. ``content`` is either a plain string or a
        # nested list of blocks — recurse.
        return _extract_text(block.get("content"))

    if btype == "document":
        direct = block.get("text", "")
        if isinstance(direct, str) and direct:
            return direct
        source = block.get("source")
        if isinstance(source, dict):
            data = source.get("data")
            if isinstance(data, str):
                return data
        return _extract_text(block.get("content"))

    # image, thinking, cache_control, tool_use_id-only shells, and any
    # unknown future block types contribute no compressible text.
    return ""


def _serialize(value: Any) -> str:
    """Turn a tool-input value into a compressible string representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)
