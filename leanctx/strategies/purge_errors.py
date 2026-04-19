"""PurgeErrorsStrategy — trim content of old errored messages.

A failed tool call in turn 3 is typically useful context for a few turns
afterwards, then becomes dead weight: the error was handled, the input
that triggered it may be very large (whole file contents, giant diff,
etc.), and neither is going to be referenced again.

This strategy replaces the content of errored messages older than
``after_turns`` ago with a short placeholder. The fact of the error
survives; the bulk doesn't.
"""

from __future__ import annotations

from typing import Any

from leanctx.classifier import classify
from leanctx.compressors import ContentType

_DEFAULT_PLACEHOLDER = "[errored output purged for context compaction]"


class PurgeErrorsStrategy:
    """Replace old errored message content with a short placeholder."""

    name = "purge_errors"

    def __init__(
        self,
        after_turns: int = 4,
        placeholder: str = _DEFAULT_PLACEHOLDER,
    ) -> None:
        self.after_turns = after_turns
        self.placeholder = placeholder

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self.after_turns:
            return messages

        cutoff = len(messages) - self.after_turns
        out: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            if i < cutoff and classify(msg) == ContentType.ERROR:
                out.append({**msg, "content": self.placeholder})
            else:
                out.append(msg)
        return out
