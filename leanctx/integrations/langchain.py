"""LangChain integration — message-format conversion + runnable helper.

LangChain uses ``BaseMessage`` subclasses (``HumanMessage``,
``AIMessage``, ``SystemMessage``, ``ToolMessage``) for chat messages;
leanctx uses plain dicts with ``role`` and ``content`` keys. These
helpers bridge the two.

The :func:`to_dicts` direction is duck-typed — it reads ``.type`` and
``.content`` attributes without importing LangChain, so it works without
``langchain_core`` installed. :func:`from_dicts` and
:func:`compress_runnable` both require ``langchain_core``.

**Manual usage (any LangChain version)::**

    from langchain_core.messages import HumanMessage
    from leanctx import Middleware
    from leanctx.integrations.langchain import from_dicts, to_dicts

    messages = [HumanMessage(content="a long RAG context ...")]

    mw = Middleware({"mode": "on", "routing": {"prose": "lingua"}})
    compressed_dicts, stats = mw.compress_messages(to_dicts(messages))
    compressed_lc = from_dicts(compressed_dicts)

**Runnable usage (LCEL pipelines, v0.2):**

    from langchain_anthropic import ChatAnthropic
    from leanctx.integrations.langchain import compress_runnable

    chain = compress_runnable({
        "mode": "on",
        "routing": {"prose": "lingua"},
    }) | ChatAnthropic(model="claude-sonnet-4-6")

    response = chain.invoke([HumanMessage("... long RAG context ...")])

The runnable is a drop-in for any LangChain chat model (ChatAnthropic,
ChatOpenAI, ChatGoogleGenerativeAI, etc.) since it operates on the
provider-neutral ``BaseMessage`` layer.
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

    Tool-use metadata is preserved where present:

    * ``tool_call_id`` on ``ToolMessage`` — needed for the result to
      link back to the matching ``tool_calls`` entry on the preceding
      assistant message. Round-trips via :func:`from_dicts`.
    * ``name`` on ``FunctionMessage`` / ``ToolMessage`` — optional.

    ``AIMessage.tool_calls`` round-tripping is a known v0.2 gap —
    preserved here as a best-effort extra key; :func:`from_dicts`
    currently reconstructs a plain ``AIMessage`` without them.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        mtype = getattr(m, "type", None) or "user"
        role = _LC_TYPE_TO_ROLE.get(mtype, mtype)
        d: dict[str, Any] = {
            "role": role,
            "content": getattr(m, "content", ""),
        }
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            d["tool_call_id"] = tool_call_id
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = tool_calls
        out.append(d)
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
            out.append(cls(content=content, tool_call_id=d.get("tool_call_id", "")))
        elif cls is AIMessage:
            # Round-trip tool_calls when present so assistant->tool linkage
            # survives. LangChain's AIMessage accepts tool_calls: list of
            # {name, args, id} dicts.
            ai_kwargs: dict[str, Any] = {"content": content}
            if "tool_calls" in d:
                ai_kwargs["tool_calls"] = d["tool_calls"]
            out.append(cls(**ai_kwargs))
        else:
            out.append(cls(content=content))
    return out


_RUNNABLE_INSTALL_HINT = (
    "langchain_core is required for compress_runnable(). "
    "Install with: pip install langchain_core"
)


def compress_runnable(
    config: dict[str, Any] | None = None,
) -> Any:
    """Return a LangChain ``Runnable`` that compresses a message list.

    The returned runnable takes ``list[BaseMessage]`` and returns
    ``list[BaseMessage]`` with the middleware applied. Compose it with
    any LangChain chat model via ``|``::

        chain = compress_runnable({"mode": "on"}) | ChatAnthropic(...)
        response = chain.invoke([HumanMessage("...")])

    One ``Middleware`` is constructed per call to ``compress_runnable``
    and reused across invocations of the resulting runnable, so
    per-Middleware state (Strategies like ``DedupStrategy``) persists
    across chain invocations. Instantiate a fresh runnable per logical
    session if you need scoped dedup.

    Requires ``langchain_core``.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as e:
        raise ImportError(_RUNNABLE_INSTALL_HINT) from e

    # Local import to avoid a top-level cycle — client.py imports from
    # leanctx.integrations.langchain wouldn't happen, but middleware.py
    # could if we ever restructure.
    from leanctx.middleware import Middleware

    mw = Middleware(config or {})

    def _compress(messages: list[Any]) -> list[Any]:
        compressed_dicts, _ = mw.compress_messages(to_dicts(messages))
        return from_dicts(compressed_dicts)

    return RunnableLambda(_compress)
