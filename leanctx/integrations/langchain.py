"""LangChain integration — message format conversion.

LangChain uses ``BaseMessage`` subclasses (``HumanMessage``,
``AIMessage``, ``SystemMessage``, ``ToolMessage``) for chat messages;
leanctx uses plain dicts with ``role`` and ``content`` keys. These
helpers bridge the two.

The :func:`to_dicts` direction is duck-typed — it reads ``.type`` and
``.content`` attributes without importing LangChain, so it works without
``langchain_core`` installed. :func:`from_dicts` requires
``langchain_core`` because it constructs concrete ``BaseMessage``
instances.

Usage::

    from langchain_core.messages import HumanMessage
    from leanctx import Middleware
    from leanctx.integrations.langchain import from_dicts, to_dicts

    messages = [HumanMessage(content="a long RAG context ...")]

    mw = Middleware({"mode": "on", "routing": {"prose": "lingua"}})
    compressed_dicts, stats = mw.compress_messages(to_dicts(messages))
    compressed_lc = from_dicts(compressed_dicts)

    # Use compressed_lc with any LangChain chat model.

v0.2 will ship ``LeanctxChatAnthropic`` — a ``ChatAnthropic`` subclass
that does this conversion + compression automatically.
"""

from __future__ import annotations

from typing import Any

_LC_INSTALL_HINT = (
    "langchain_core is required for from_dicts(). "
    "Install with: pip install langchain_core"
)

# LangChain BaseMessage.type values -> our role labels.
_LC_TYPE_TO_ROLE = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}

_ROLE_TO_LC_TYPE = {v: k for k, v in _LC_TYPE_TO_ROLE.items()}


def to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain ``BaseMessage`` objects to leanctx dicts.

    Duck-typed: reads ``msg.type`` and ``msg.content`` without importing
    LangChain. Unknown types pass through as their own role string so
    users can experiment with custom message types.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        mtype = getattr(m, "type", None) or "user"
        role = _LC_TYPE_TO_ROLE.get(mtype, mtype)
        content = getattr(m, "content", "")
        out.append({"role": role, "content": content})
    return out


def from_dicts(dicts: list[dict[str, Any]]) -> list[Any]:
    """Convert leanctx dicts back to LangChain ``BaseMessage`` objects.

    Requires ``langchain_core``. Raises :class:`ImportError` with a clear
    install hint if it's missing.
    """
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
    except ImportError as e:
        raise ImportError(_LC_INSTALL_HINT) from e

    role_map: dict[str, Any] = {
        "user": HumanMessage,
        "human": HumanMessage,
        "assistant": AIMessage,
        "ai": AIMessage,
        "system": SystemMessage,
        "tool": ToolMessage,
    }

    out: list[Any] = []
    for d in dicts:
        role = d.get("role", "user")
        cls = role_map.get(role, HumanMessage)
        content = d.get("content", "")
        if cls is ToolMessage:
            # ToolMessage needs tool_call_id; fall back to empty string
            # when the dict doesn't carry one. v0.2 will preserve
            # tool_call_id through compression round-trips.
            out.append(cls(content=content, tool_call_id=d.get("tool_call_id", "")))
        else:
            out.append(cls(content=content))
    return out
