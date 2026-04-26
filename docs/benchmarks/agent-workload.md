# Coding-agent workload: end-to-end compression

**Date:** 2026-04-25
**Reproducible via:** [`scripts/integration_test_agent_workload.py`](../../scripts/integration_test_agent_workload.py)

## Why this benchmark exists

The [SelfLLM provider comparison](selfllm-providers.md) used a prose-heavy SRE incident document. That's a fair compression test for RAG-style traffic, but it understates the case for users running coding agents.

The interesting workload for leanctx is **agent traffic**:

- Tool outputs (file contents, grep results, log dumps, error traces) — typically 60-90% of the input tokens
- Code blocks that *must not* be rewritten (compressing a closing brace or rewriting an `if` would silently corrupt code)
- `tool_use` ↔ `tool_result` linkage that *must* survive compression
- Stack traces and errors that are diagnostic-critical and shouldn't be touched

This benchmark exercises a realistic 9-message agent transcript end-to-end through `leanctx.Anthropic` with the full content-aware routing config and asserts that all four invariants hold.

## TL;DR

```
Original:      7898 chars, ~2148 tokens
Sent on wire:  5701 chars, ~1384 tokens
Reduction:    27.8% chars, 35.6% tokens
Saved:                       768 tokens per request
Method:                      hybrid (Lingua + Verbatim)
```

**The 35.6% token reduction came entirely from prose / log-output content. Every code block, every error, every tool-call argument was preserved byte-identically.**

## Methodology

### Sample transcript

A 9-message debugging session that mirrors what a coding agent like Claude Code, Cursor, or a custom LangGraph agent produces:

| # | Role | Content | Compressible? |
|:-:|---|---|:-:|
| 0 | user | "Production auth is rejecting valid JWTs..." | yes (prose) |
| 1 | assistant | reasoning + `tool_use(read_file)` | partial (text yes, tool_use no) |
| 2 | user | `tool_result` containing 2 KB Python source | **no — code, must be verbatim** |
| 3 | assistant | reasoning + `tool_use(grep)` | partial |
| 4 | user | `tool_result` with grep output | partial |
| 5 | assistant | reasoning + `tool_use(run_command)` for logs | partial |
| 6 | user | `tool_result` with 3.4 KB of repetitive log lines | **yes — prime target** |
| 7 | assistant | reasoning + `tool_use(edit_file)` | partial (input.new_str must be exact) |
| 8 | user | `tool_result` with `is_error: True`, Python traceback | **no — error, must be verbatim** |

Total: 7,898 chars, ~2,148 tokens. The full source is in [`scripts/integration_test_agent_workload.py`](../../scripts/integration_test_agent_workload.py).

### Configuration

```python
client = Anthropic(leanctx_config={
    "mode": "on",
    "trigger": {"threshold_tokens": 0},
    "routing": {
        "code":           "verbatim",
        "error":          "verbatim",
        "prose":          "lingua",
        "long_important": "lingua",
    },
    "lingua": {"ratio": 0.5},
})
```

`Lingua` runs the real Microsoft LLMLingua-2 model locally on Apple Silicon MPS. Anthropic's API endpoint is mocked via `respx` so no API key is required to run the benchmark.

## Per-message results

```
 #  role         before   after       Δ  content type
 0  user           185     101      84   text
 1  assistant      237     154      83   text + tool_use
 2  user          2102    2102       0   tool_result (Python source)    ← verbatim
 3  assistant      327     237      90   text + tool_use
 4  user           691     386     305   tool_result (grep output)
 5  assistant      289     203      86   text + tool_use
 6  user          3423    1902    1521   tool_result (log dump)         ← 45% compressed
 7  assistant      288     260      28   text + tool_use
 8  user           356     356       0   tool_result (error)            ← verbatim
```

The big wins:

- **Message 6 (log dump):** 3,423 → 1,902 chars = 44.4% reduction. Repetitive log lines compress beautifully.
- **Message 4 (grep output):** 691 → 386 chars = 44.1% reduction. Prose-style narration around the grep results gets summarized.
- **Message 0 (user prompt):** 185 → 101 chars = 45.4% reduction. The user's question got summarized to its essential ask.

The deliberate non-wins:

- **Message 2 (Python source):** 2,102 chars in, 2,102 chars out, zero change. The classifier detected code (multiple `def` / `class` / `import` lines) and routed to Verbatim. Compressing this would risk dropping a closing bracket or rewriting an `if` and silently corrupting the file the agent is supposed to edit.
- **Message 8 (error):** 356 chars in, 356 chars out. Stack traces are diagnostic-critical; rewriting them strips the very information the agent is trying to use.

## Correctness assertions

The script asserts five invariants programmatically — any of these failing should fail the benchmark:

```
✅ tool_use_id linkage preserved: 8 ids matched between tool_use blocks
   and tool_result blocks before/after compression. The conversation's
   tool-call graph is structurally identical.

✅ Code block preserved verbatim: src/auth/middleware.py contents are
   byte-identical pre-compression vs post-compression.

✅ Error block preserved verbatim: the FileChangedError traceback is
   byte-identical.

✅ tool_use input fields preserved: edit_file's new_str and grep's
   pattern argument both match the originals exactly. Compressing
   these would change the actual tool invocation, which would break
   what the agent is trying to do.

✅ Log output was actually compressed: 3423 → 1902 chars. The classifier
   correctly identified this as compressible content (not code, not error)
   and the router sent it through Lingua.
```

If any of those failed in the future, the benchmark exits non-zero — making this a regression test for the structural-integrity guarantees of the v0.2 block-aware compressor.

## What this means in dollars

Assume a coding agent that averages 2,000 input tokens per request (this transcript was 2,148 — representative). At 35.6% reduction:

| Underlying model | Input rate | Daily savings @ 100K req/day |
|---|:-:|:-:|
| Claude Sonnet 4.6 | $3 / MT | **$210/day**, **$76,650/year** |
| Claude Opus 4.7   | $15 / MT | **$1,050/day**, **$383,250/year** |
| GPT-5             | $5 / MT  | **$350/day**, **$127,750/year** |
| Gemini 2.5 Pro    | $1.25 / MT | **$87/day**, **$31,755/year** |

Subtract the cost of compression itself. With `Lingua` (local model), that's $0 marginal. With `SelfLLM` on the cheapest tier, it's at most a few dollars per million requests.

The break-even is essentially immediate.

## Caveats

- **Single transcript.** Real agent traffic is more diverse — some sessions are 80% code-reading (low compression headroom), others are 80% summarizing logs (high). The 35.6% number is for a session with mixed content.
- **Lingua compresses prose, not natural-language code comments.** A docstring-heavy file might shrink slightly even under Verbatim if the classifier disagrees. The current classifier is conservative and routes anything that smells like code to Verbatim.
- **The benchmark uses an English log dump.** Non-English content compresses at a lower ratio; the LLMLingua-2 model is multilingual but tuned on English.
- **Numbers measured locally on Apple Silicon MPS.** A CUDA box would be faster on the model load but identical on compression ratio (the ratio is a property of the model, not the device).

## Reproducing

```bash
pip install 'leanctx[lingua,anthropic,bench]'
python scripts/integration_test_agent_workload.py
```

The `bench` extra adds `respx`, which mocks the Anthropic HTTP endpoint so this script needs no API key. First run downloads ~1.2 GB of model weights to `~/.cache/huggingface/`; subsequent runs use the cached model and complete in <30s end-to-end. The script exits non-zero on any structural-integrity regression (mutated code block, mutated error, broken tool linkage, mutated tool input, or log payload that fails to compress).

## See also

- [SelfLLM cross-provider comparison](selfllm-providers.md) — same compressor architecture, but using cloud LLMs (Anthropic / OpenAI / Gemini) as the summarizer instead of a local model.
