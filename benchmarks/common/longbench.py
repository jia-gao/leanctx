"""LongBench v2 loading, prompt rendering, and answer extraction (extracted
from bench_phase1). Provider-agnostic — the eval call lives in the runner.
"""
from __future__ import annotations

import re
from typing import Any

_LB_PROMPT_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n$DOC$\n</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n"
    "You must commit to exactly one option even if the text is thin or missing: "
    "pick the single most likely choice from the text and your own best guess, "
    "and never reply that you cannot answer.\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)


_LB_ANS_RE_PAREN = re.compile(r"The correct answer is \(([A-D])\)")


_LB_ANS_RE_BARE = re.compile(r"The correct answer is ([A-D])")


def _build_lb_prompt(item: dict, context: str) -> str:
    """Render the LongBench MC prompt for an item against a context string.

    The same template is used for every leg (open- and closed-book): the only
    thing that varies is ``context`` (the ``$DOC$`` slot), so any accuracy gap
    is attributable to the document, not to a prompt difference.
    """
    return (
        _LB_PROMPT_TEMPLATE
        .replace("$DOC$", context)
        .replace("$Q$", item.get("question", ""))
        .replace("$C_A$", item.get("choice_A", ""))
        .replace("$C_B$", item.get("choice_B", ""))
        .replace("$C_C$", item.get("choice_C", ""))
        .replace("$C_D$", item.get("choice_D", ""))
    )


def _extract_lb_answer(response: str) -> str | None:
    m = _LB_ANS_RE_PAREN.search(response.replace("*", ""))
    if not m:
        m = _LB_ANS_RE_BARE.search(response)
    return m.group(1) if m else None


def _lb_head_tail(text: str, max_chars: int = 112_000) -> str:
    """Truncate to ~28K tokens (112K chars ≈ 4 chars/token) to fit 30K tok/min rate limit."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _load_lb_items(
    limit: int = 5,
    workload_tag: str = "lb_s1",
    *,
    seed: int = 1234,
    oversample_long: bool = True,
) -> list[dict[str, Any]]:
    """Load and format LongBench v2 items for run_leg().

    Sampling is stratified across (length × difficulty) cells but **random
    within each cell** (seeded for reproducibility) — the previous
    ``cell_items[:per_cell]`` took the first-N of each cell, which produced
    the suspiciously balanced buckets the reviewer called out.

    With ``oversample_long`` the ``long`` length category is given 1.5×
    weight, since that is where Layer 8 showed a real (−10%) accuracy risk at
    small N and needs more samples before any safety claim.
    """
    import random as _random

    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench-v2", split="train")
    items: list[dict] = list(ds)

    if limit > 0 and len(items) > limit:
        rng = _random.Random(seed)
        cells: dict[tuple[str, str], list[dict]] = {}
        for it in items:
            key = (it.get("length", ""), it.get("difficulty", ""))
            cells.setdefault(key, []).append(it)

        # Weight cells; long-context cells get 1.5× weight when oversampling.
        weights = {
            key: (1.5 if oversample_long and key[0] == "long" else 1.0)
            for key in cells
        }
        total_weight = sum(weights.values())

        sample: list[dict] = []
        for key, cell_items in cells.items():
            quota = max(1, round(limit * weights[key] / total_weight))
            quota = min(quota, len(cell_items))
            sample.extend(rng.sample(cell_items, quota))

        # Trim/pad to exactly `limit` from the remaining pool, randomly.
        rng.shuffle(sample)
        if len(sample) > limit:
            sample = sample[:limit]
        elif len(sample) < limit:
            chosen = {id(x) for x in sample}
            remaining = [it for it in items if id(it) not in chosen]
            rng.shuffle(remaining)
            sample.extend(remaining[: limit - len(sample)])
        items = sample

    result = []
    for i, it in enumerate(items):
        context = _lb_head_tail(it["context"])
        result.append({
            "messages": [{"role": "user", "content": context}],
            "workload": workload_tag,
            "item_id": it.get("_id", f"lb_{i:04d}"),
            "question": it["question"],
            "choice_A": it["choice_A"],
            "choice_B": it["choice_B"],
            "choice_C": it["choice_C"],
            "choice_D": it["choice_D"],
            "gold": it["answer"],
            # metadata threaded into records for accuracy breakdown tables
            "lb_domain": it.get("domain", ""),
            "lb_difficulty": it.get("difficulty", ""),
            "lb_length": it.get("length", ""),
        })
    return result
