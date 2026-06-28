"""Coding-agent transcript loading for the InsForge savings workload.

Two sources (see the design doc §5):

* ``agent_extended`` — leanctx's shipped 50-turn synthetic agent workload. Free,
  deterministic; the CI/dev fixture.
* SWE-bench Verified agent **trajectories** (``swe-bench/experiments`` ``trajs/``
  dumps) — real coding-agent traffic; the headline. ``swebench_traj_to_messages``
  normalizes the common ``.traj`` shapes into an OpenAI message list.

A loaded transcript is split into **items** (one per episode / chunk) so the
savings run produces several data points rather than a single number.
"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

_OPENAI_ROLES = {"system", "user", "assistant", "tool", "function"}


def _stringify(content: Any) -> str | None:
    """Best-effort flatten of a message ``content`` to a string, or None."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
        return "\n".join(parts) if parts else None
    return None


def swebench_traj_to_messages(traj: Any) -> list[dict[str, Any]]:
    """Normalize a SWE-bench / SWE-agent trajectory into OpenAI messages.

    Tolerant of the common shapes seen in ``swe-bench/experiments`` ``trajs/``:
      * a bare list of ``{role, content}``
      * ``{"history": [...]}`` (SWE-agent ``.traj``)
      * ``{"trajectory": [{"messages": [...]}, ...]}`` or a list of steps each
        carrying ``messages`` / ``response`` / ``observation``.

    Roles outside the OpenAI set are mapped to ``user`` (observations) or
    ``assistant`` (model output); non-string content is flattened.
    """
    raw: list[Any]
    if isinstance(traj, dict):
        if isinstance(traj.get("history"), list):
            raw = traj["history"]
        elif isinstance(traj.get("messages"), list):
            raw = traj["messages"]
        elif isinstance(traj.get("trajectory"), list):
            raw = []
            for step in traj["trajectory"]:
                if isinstance(step, dict):
                    if isinstance(step.get("messages"), list):
                        raw.extend(step["messages"])
                    else:
                        raw.append(step)
                else:
                    raw.append(step)
        else:
            raw = []
    elif isinstance(traj, list):
        raw = traj
    else:
        raw = []

    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = _stringify(entry.get("content"))
        if content is None:
            # SWE-agent step records: response = model text, observation = tool out
            if isinstance(entry.get("response"), str):
                out.append({"role": "assistant", "content": entry["response"]})
            if isinstance(entry.get("observation"), str):
                out.append({"role": "user", "content": entry["observation"]})
            continue
        if role not in _OPENAI_ROLES:
            role = "assistant" if role in ("ai", "model", "bot") else "user"
        out.append({"role": role, "content": content})
    return out


def _split_into(messages: list[dict[str, Any]], episodes: int | None) -> list[list[dict[str, Any]]]:
    """Split a transcript into ``episodes`` contiguous items, or one full item.

    ``episodes`` is the number of *items* (data points), not a turn count. The
    default (None) returns a single full-context item — how a coding agent's
    request actually looks (the whole conversation so far), and large enough to
    clear the sidecar's per-request token gate. Splitting is for big real
    trajectories where each slice is still over the gate.
    """
    if not messages:
        return []
    if episodes is None or episodes <= 1:
        return [messages]
    size = ceil(len(messages) / episodes)
    return [messages[i : i + size] for i in range(0, len(messages), size)]


def load_agent_transcript(
    *,
    episodes: int | None = None,
    turns: int | None = None,
) -> list[list[dict[str, Any]]]:
    """leanctx's ``agent_extended`` as transcript item(s).

    ``turns`` caps the total message count first; ``episodes`` splits the result
    into that many contiguous items (default: one full-context item).
    """
    from leanctx.bench.workloads import load_workload

    msgs = load_workload("agent_extended")
    if turns is not None:
        msgs = msgs[:turns]
    return _split_into(msgs, episodes)


def load_swebench_trajectory(
    path: str | Path,
    *,
    episodes: int | None = None,
    turns: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Load a SWE-bench Verified ``.traj``/``.json`` file into transcript item(s)."""
    data = json.loads(Path(path).read_text())
    msgs = swebench_traj_to_messages(data)
    if turns is not None:
        msgs = msgs[:turns]
    return _split_into(msgs, episodes)


def load_transcript_items(
    source: str = "agent_extended",
    *,
    episodes: int | None = None,
    turns: int | None = None,
) -> list[dict[str, Any]]:
    """Return benchmark items (one per episode) for the savings workload.

    ``source`` is ``"agent_extended"`` or a path to a SWE-bench trajectory file.
    Each item is ``{messages, workload, item_id}`` — no gold (savings-only).
    """
    if source == "agent_extended":
        chunks = load_agent_transcript(episodes=episodes, turns=turns)
    else:
        chunks = load_swebench_trajectory(source, episodes=episodes, turns=turns)
    label = Path(source).stem if source != "agent_extended" else "agent_extended"
    return [
        {
            "messages": chunk,
            "workload": "transcript",
            "item_id": f"transcript::{label}::ep{i}",
        }
        for i, chunk in enumerate(chunks)
        if chunk
    ]
