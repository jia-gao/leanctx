"""Shared helpers for reading chat-message content.

OpenAI-style messages carry a plain string in ``content``. Anthropic-style
messages may carry a list of typed blocks — text, images, tool_use, etc.
Both shapes flow through leanctx, so any module that wants to inspect the
text needs to normalize first. This module centralizes that logic.
"""

from __future__ import annotations

from typing import Any


def get_text_content(message: dict[str, Any]) -> str:
    """Return the concatenated text from a chat message.

    * String content (OpenAI-style) is returned as-is.
    * List content (Anthropic-style): text blocks are joined with newlines;
      non-text blocks (images, tool_use, tool_result) are dropped.
    * Anything else returns an empty string.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts)
    return ""
