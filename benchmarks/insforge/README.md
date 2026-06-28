# InsForge × leanctx — Measurement Harness (I3)

Benchmarks leanctx prompt compression on **InsForge's** chat path with compression **off vs on**:

- **(a) LongBench v2** — accuracy (does compression preserve answer quality?), and the
  workload the headline savings + Go/No-Go gate are computed on
- **(b) a coding-agent transcript** — token savings (optional; off by default). It still runs
  and is recorded, but is **informational only**: it is *excluded* from the gate/headline so the
  verdict never blends two workloads with very different compressibility

It reports savings on **InsForge's own `usage.prompt_tokens`** and dollars via **OpenRouter pricing**,
reusing the provider-agnostic core in `benchmarks/common/` (the same scorer/LongBench/verbatim/report
logic as the ClawRouter `bench_phase1.py`). Full design + results: `research/insforge/I3-measurement-harness-design-and-plan.md`.

Entry point: `python -m benchmarks.insforge.bench_insforge [options]`

---

## TL;DR (copy-paste)

```bash
# 0) one-time: install deps (GPU box recommended for the Lingua sidecar)
pip install -e ".[all,dev]"          # includes openai, anthropic, llmlingua, tiktoken, fastapi, datasets
# put keys in a .env at the repo root (auto-loaded), or export them:
#   OPENROUTER_API_KEY=sk-or-...      (for --upstream openrouter)
#   ANTHROPIC_API_KEY=sk-ant-...      (for --upstream anthropic)

# 1) FREE smoke — no API, no InsForge stack; validates sidecar + report wiring
python -m benchmarks.insforge.bench_insforge --dry-run --lb-n 2 \
    --out /tmp/r.jsonl --report /tmp/r.md

# 2) DIRECT run (recommended for the rich report) — OpenRouter → Claude Haiku, 50 LongBench samples
python -m benchmarks.insforge.bench_insforge --upstream openrouter \
    --model anthropic/claude-haiku-4.5 --lb-n 50 --lingua-device cuda \
    --out benchmarks/insforge/results/run.jsonl \
    --report benchmarks/insforge/results/run.md

# 3) GATEWAY run (literal "through InsForge") — clones+builds+runs InsForge in Docker
python -m benchmarks.insforge.bench_insforge --upstream insforge \
    --model anthropic/claude-haiku-4.5 --lb-n 10 --lingua-device cuda \
    --out benchmarks/insforge/results/gw.jsonl \
    --report benchmarks/insforge/results/gw.md
```

---

## Two routing modes — pick one with `--upstream`

| `--upstream` | What happens | Measures | Needs |
|---|---|---|---|
| `openrouter` *(direct)* | Harness compresses via the sidecar, then calls OpenRouter directly | `usage.prompt_tokens` from OpenRouter | `OPENROUTER_API_KEY` + GPU sidecar |
| `anthropic` *(direct)* | Same, but calls the Anthropic API (proxy when no OpenRouter key) | Anthropic `usage.input_tokens` | `ANTHROPIC_API_KEY` + GPU sidecar |
| `openai` *(direct)* | Same, via OpenAI | OpenAI `usage.prompt_tokens` | `OPENAI_API_KEY` + GPU sidecar |
| `insforge` *(gateway, default)* | Clones+patches+builds+runs **InsForge v2.2.2** in Docker; toggles compression OFF→restart→ON inside the gateway | **InsForge's own `usage.prompt_tokens`** | Docker + Node ≥20 + GPU sidecar; `OPENROUTER_API_KEY` (set in InsForge's env) |

### Which mode should I use?

- **Want the full ClawRouter-style report** (verbatim-vs-savings route split, by difficulty/length/domain,
  §2b verbatim-excluded)? → **direct** (`--upstream openrouter`). The route split is only observable when the
  **harness** performs the compression. Direct savings is **byte-identical** to the gateway number (the gateway
  is a passthrough to OpenRouter), so this is the recommended way to get the rich report.
- **Want the literal "through InsForge's gateway" number** (InsForge's own `usage`, server-side compression
  via the inlined connector)? → **gateway** (`--upstream insforge`). Note: in gateway mode the per-message
  routing is **not** observable, so §2b and the by-route rows show **N/A**; the headline savings still comes
  from InsForge's `usage.prompt_tokens`.

---

## Prerequisites

**Always:**
- Python deps: `pip install -e ".[all,dev]"` (key extras: `openai`, `anthropic`, `lingua`=llmlingua,
  `tokens`=tiktoken, `server`=fastapi/uvicorn, `longbench`=datasets; `dev` brings `python-dotenv`).
  The harness **hard-fails without tiktoken** (token counts must be exact, not char/4).
- A **GPU** for the Lingua sidecar (CPU works but is multi-second/call). Set `--lingua-device cuda`
  (RTX 50xx / Blackwell needs a recent torch/CUDA build). The sidecar is `leanctx-serve`, spawned
  automatically; the server logs the resolved device — confirm it says `cuda`.

**For real (non-`--dry-run`) runs:** an API key for the chosen provider, in a repo-root `.env`
(auto-loaded) or the shell env.

**For `--upstream insforge` (gateway) only:**
- **Docker + Docker Compose** and **Node ≥ 20** (InsForge builds from source).
- `OPENROUTER_API_KEY` — written into InsForge's env so its gateway can reach OpenRouter (a paid
  OpenRouter account; the free tier caps prompt tokens ~21.6K, below LongBench's ~30K docs).
- The harness binds the sidecar to `0.0.0.0` automatically so the InsForge container can reach it via
  `host.docker.internal`.

---

## All options

| Flag | Default | Meaning |
|---|---|---|
| `--upstream {insforge,openrouter,anthropic,openai}` | `insforge` | Routing mode (see table above). |
| `--model MODEL` | `anthropic/claude-haiku-4.5` | OpenRouter/InsForge model slug, or provider-native id for `--upstream anthropic` (e.g. `claude-haiku-4-5-20251001`). |
| `--lb-n N` | `10` | Number of LongBench v2 questions. `0` disables LongBench. Full set = `503`. |
| `--transcript` | off | Also run the coding-agent transcript savings workload (informational; **excluded** from the headline + gate). |
| `--transcript-source S` | `agent_extended` | `agent_extended` (shipped) or a path to a SWE-bench `.traj`/`.json`. |
| `--transcript-episodes N` | `1` | Split the transcript into N contiguous items (1 = one full-context item). |
| `--transcript-turns N` | none | Cap the transcript to the first N messages. |
| `--sidecar-url URL` | `http://127.0.0.1:8459` | leanctx sidecar URL (auto-spawned if not already healthy). |
| `--lingua-device D` | `cuda` | `auto`\|`cuda`\|`cpu`\|`mps` for the sidecar. |
| `--lingua-ratio R` | `0.5` | Fraction of prose tokens to **keep** (lower = more aggressive). |
| `--closed-book` / `--no-closed-book` | on | Run the no-document control leg (rules out "context was irrelevant"). |
| `--dry-run` | off | No provider/gateway calls — sidecar + tiktoken only. Free wiring check. |
| `--lb-max-tokens N` | `1024` | Eval `max_tokens`. **Keep generous** — a small budget truncates the model's answer line on long docs → unparseable → false 0% accuracy. |
| `--sample-seed N` | `1234` | Seed for the stratified LongBench subset (reproducible). |
| `--no-oversample-long` | oversample on | Disable 1.5× weighting of long-context items in subsets. |
| `--savings-threshold F` | `0.20` | Gate: minimum Δ prompt-token savings. |
| `--accuracy-drop F` | `0.02` | Gate: max allowed accuracy drop. |
| `--refresh-pricing` | off | Pull live OpenRouter prices instead of the pinned table. |
| `--insforge-ref REF` | `v2.2.2` | InsForge git ref to clone/patch/build (gateway mode). |
| `--insforge-workdir DIR` | `/tmp/insforge_bench` | Where InsForge is cloned + run (gateway mode). |
| `--skip-insforge-setup` | off | Reuse an existing clone/patch/.env in the workdir (gateway mode). |
| `--gateway-url URL` | `http://localhost:7130` | InsForge gateway base URL (gateway mode). |
| `--gateway-key KEY` | none | Use a pre-minted InsForge user JWT instead of register/login bootstrap. |
| `--out PATH` | `./insforge_results.jsonl` | Per-record JSONL output. |
| `--report PATH` | `./insforge_report.md` | Markdown report output. |

Exit code: `0` = PASS, `1` = NO-GO (gate failed), `2` = fatal (e.g. tiktoken missing),
`3` = methodology self-check failed (artifacts still written; metrics untrustworthy).

---

## Outputs

Written next to `--out`:
- `<report>.md` — the report. Sections mirror the ClawRouter report
  (`benchmarks/clawrouter/full_long_bench_evaluation_result.md`):
  1. Go/No-Go Gate · 2. Token Compression (`usage.prompt_tokens`) + tiktoken cross-check ·
  2b. Compression Excluding Verbatim (direct only) · 3. LongBench Accuracy — overall + closed-book +
  by leanctx route + by difficulty/length/domain (route-split) · 4. Latency · 5. Cost.
- `<out>.jsonl` — one record per item per leg (`OFF`/`ON`/`CB`), incl. `usage_prompt_tokens`, `accuracy`,
  `lb_domain/difficulty/length`, `lx_route` (direct).
- `<out dir>/by_item/` — per-item deep dives (full messages + all fields).

### I4 savings-vs-mix curve (optional)

```bash
python benchmarks/clawrouter/reweight_prose_code_curve.py \
    --data benchmarks/insforge/results/run.jsonl \
    --results benchmarks/insforge/results --leg ON --workload lb_if   # needs matplotlib
```

---

## Cost & time (rough, Haiku 4.5, direct)

| `--lb-n` | API cost | Wall time |
|---|---|---|
| 2 (smoke) | ~$0.02 | ~1 min |
| 10 | ~$0.10 | ~3 min |
| 50 | ~$0.5 | ~12 min |
| 503 (full) | ~$15–20 | ~1.5–2 h (sequential) |

`--dry-run` is free. Savings legs use the provider's own counts; LongBench OFF sends full ~30K-token docs.

---

## Tests

```bash
pytest -m unit       tests/insforge/          # pure logic: pricing, transcript, patch, scoring (no I/O)
pytest -m insforge   tests/integration/test_insforge_sidecar.py   # GPU sidecar, no API/Node
pytest -m e2e        tests/e2e/test_insforge_e2e.py                # small live subset (needs a key)
```

---

## Troubleshooting

- **`ModuleNotFoundError: openai`** — install it: `pip install -e ".[openai]"` (needed for openrouter/openai/insforge).
- **OpenRouter `402 Insufficient credits` / `Prompt tokens limit exceeded`** — the account has no credits or
  is on the free tier (caps prompt tokens ~21.6K < LongBench's ~30K). Add credits / use a paid account, or
  test with the transcript workload (`--lb-n 0 --transcript`, ~3K tokens).
- **All LongBench predictions `None` / 0% accuracy** — `--lb-max-tokens` too small; keep it ≥ 1024.
- **Accuracy gate NO-GO at small N** — at `--lb-n 10` accuracy is noise-dominated; check the closed-book
  control (if open-book ≤ closed-book, context isn't carrying the answer). Use `--lb-n 50`+ for a real verdict.
- **Gateway: container can't reach the sidecar** — the harness binds the sidecar to `0.0.0.0` and adds
  `host.docker.internal:host-gateway`; ensure nothing else holds port 8459.
- **Gateway is slow to build** — first `docker compose up --build` pulls Postgres/Deno + compiles the InsForge
  backend (several minutes); subsequent runs with `--skip-insforge-setup` reuse the cache.
- **Tear down the gateway stack:** `docker compose -f /tmp/insforge_bench/docker-compose.yml down -v`.

---

## Notes for Claude Code

- Long runs (`--lb-n 50`/`503`) are sequential — launch them as **background** Bash tasks and wait for the
  completion notification; poll progress via `/compress` count in the log (ON-leg items) or `by_item/` (written
  only at the end).
- **Gateway is NOT OpenAI-compatible** (issue #11): `POST /api/ai/chat/completion`, `verifyUser` JWT,
  `{model, messages, maxTokens}` body, usage at `metadata.usage.promptTokens` — the harness handles this; do
  not point an OpenAI client at `/v1`.
- The connector is **inlined** into InsForge's patched `chat-completion.service.ts` (no npm dep) so it survives
  the in-image `npm ci`.
- Default `--upstream insforge` requires Docker; for a quick number without Docker use `--upstream openrouter`.
