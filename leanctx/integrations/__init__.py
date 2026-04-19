"""Framework integrations.

v0.1 starter integrations — small, pragmatic helpers that let users wire
leanctx into LangChain, LlamaIndex, etc. without leanctx taking on hard
dependencies on those frameworks.

Currently available:

* :mod:`leanctx.integrations.langchain` — message-format conversion
  between LangChain ``BaseMessage`` and leanctx dict shape

Full drop-in wrappers (e.g. a ``LeanctxChatAnthropic`` subclass of
``ChatAnthropic``) land in v0.2 once each framework's streaming /
tool-use semantics are validated.
"""

__all__: list[str] = []
