"""Adapter between Gemini's ``contents`` parameter and leanctx's
``list[dict]`` message shape.

Gemini accepts multiple input shapes:

* a plain string: ``contents="hello"``
* a list of strings: ``contents=["hi", "ho"]``
* a list of ``Content`` objects (or dicts): each has ``role`` + ``parts``
* a list mixing strings and Content objects

We support the first three via duck-typing (never imports google.genai).
``parts`` is handled for **text-only** members — any ``Part`` that's a
function_call, function_response, or image triggers an "opaque" bailout
that returns an empty message list, and the caller then skips the
middleware entirely. That preserves correctness (we never rewrite
function-call payloads) at the cost of skipping compression for those
requests. Full multimodal + function-call normalization is v0.3 work.

Gemini doesn't have a native system role; a ``role="system"`` message
going the other direction is mapped to ``role="user"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Gemini role names -> leanctx role names.
_GEMINI_TO_LEANCTX_ROLE = {
    "user": "user",
    "model": "assistant",
}

# Reverse mapping. Gemini has no "system" role — flatten to "user" on
# the way back.
_LEANCTX_TO_GEMINI_ROLE = {
    "user": "user",
    "assistant": "model",
    "system": "user",
}


@dataclass(frozen=True)
class ContentsShape:
    """Enough context to roundtrip back to the original Gemini shape."""

    kind: str  # "string" | "list" | "opaque"


OPAQUE = ContentsShape(kind="opaque")


def contents_to_messages(contents: Any) -> tuple[list[dict[str, Any]], ContentsShape]:
    """Normalize Gemini ``contents`` to leanctx messages.

    Returns ``(messages, shape)``. When ``shape.kind == "opaque"``, the
    caller should skip the middleware entirely and pass through — the
    input couldn't be normalized (non-text parts, unrecognized shape).
    """
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}], ContentsShape(kind="string")

    if isinstance(contents, list):
        messages: list[dict[str, Any]] = []
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            msg, ok = _content_object_to_message(item)
            if not ok:
                return [], OPAQUE
            messages.append(msg)
        return messages, ContentsShape(kind="list")

    # Unknown top-level shape — can't normalize.
    return [], OPAQUE


def _content_object_to_message(item: Any) -> tuple[dict[str, Any], bool]:
    """Turn a single Gemini Content (object or dict) into a leanctx
    message. Returns ``(msg, ok)`` — ``ok=False`` signals any
    non-text part, which forces the caller to bail out to passthrough.
    """
    role_raw = _get(item, "role") or "user"
    parts = _get(item, "parts") or []
    if not isinstance(parts, list):
        return {}, False

    texts: list[str] = []
    for part in parts:
        text = _get(part, "text")
        if text is None or text == "":
            # Part is not a text part (function_call, function_response,
            # image, file_data, etc.). We refuse to compress these for
            # correctness — they'd need careful preservation or
            # round-tripping that v0.2 doesn't implement yet.
            if _is_empty_text_part(part):
                # Empty-text part is benign; skip it silently.
                continue
            return {}, False
        if isinstance(text, str):
            texts.append(text)
        else:
            return {}, False

    role = _GEMINI_TO_LEANCTX_ROLE.get(str(role_raw), "user")
    return {"role": role, "content": "\n".join(texts)}, True


def _is_empty_text_part(part: Any) -> bool:
    """A part with text="" is just an empty text part — not a
    function_call or image. Distinct from parts that have no text
    attribute at all.
    """
    text = _get(part, "text")
    return bool(text == "")


def messages_to_contents(
    messages: list[dict[str, Any]], shape: ContentsShape
) -> Any:
    """Reconstruct Gemini ``contents`` from compressed leanctx messages.

    Must not be called with ``shape.kind == "opaque"`` — those never
    reach the middleware.
    """
    if shape.kind == "string":
        # Single-string input usually round-trips to a single string —
        # nicer for users who started simple.
        if len(messages) == 1:
            content = messages[0].get("content", "")
            if isinstance(content, str):
                return content
        # Something changed; fall through to list form.
        return _messages_to_content_dicts(messages)

    if shape.kind == "list":
        return _messages_to_content_dicts(messages)

    raise ValueError(
        f"opaque shape cannot be roundtripped; caller should have "
        f"skipped the middleware (got shape={shape.kind!r})"
    )


def _messages_to_content_dicts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn leanctx messages back into Gemini-style ``{role, parts}`` dicts.

    The google.genai SDK accepts dict form as well as ``Content`` objects,
    so we don't need to construct SDK types here.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = _LEANCTX_TO_GEMINI_ROLE.get(msg.get("role", "user"), "user")
        content = msg.get("content", "")
        parts = _content_to_parts(content)
        out.append({"role": role, "parts": parts})
    return out


def _content_to_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    parts.append({"text": text})
        if not parts:
            parts.append({"text": ""})
        return parts
    return [{"text": str(content)}]


def _get(obj: Any, attr: str) -> Any:
    """Duck-typed getter: works on objects (.attr) and dicts (["attr"])."""
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
