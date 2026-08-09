# ClawRouter integration — leanctx "Layer 8"

A deployable **"Layer 8"** for [ClawRouter](https://github.com/BlockRunAI/ClawRouter): a
containerized leanctx compression sidecar that adds a **semantic (LLMLingua-2) pass** on
top of ClawRouter's seven structural compression layers.

ClawRouter's `compressContext()` runs lexical/structural passes (dedup, whitespace,
dictionary, paths, JSON-compact, observation, dynamic codebook). What it intentionally
leaves untouched is the **long natural-language prose** in system/user/assistant turns —
that's exactly what leanctx compresses, while routing code, tracebacks, and structured /
tool content to **verbatim** so nothing load-bearing is dropped. The two are complementary.

```
ClawRouter request
   └─ Layers 1–7 (structural, in-process)
        └─ Layer 8  ── HTTP ──▶  leanctx sidecar (this directory)
                                   POST /compress  →  LLMLingua-2 on prose
```

This integration has two parts:

| Part | What | Status |
|------|------|--------|
| **P2a — sidecar package** | The Dockerized leanctx service here. The thing ClawRouter calls. | **here now** |
| **P2b — ClawRouter connector** | The ClawRouter-side hook that POSTs to the sidecar after Layer 7 and splices the result back. | **here now** — [`connector/`](connector/) |

> This is the *serving* sidecar. It differs from the repo-root [`Dockerfile`](../../Dockerfile),
> which is a base image that just pre-installs leanctx — this one **runs** `leanctx-serve`
> with the model baked in. Both build leanctx from the in-repo source (no PyPI release needed).

---

## Quick start (CPU)

```bash
# from the repo root
cp integrations/clawrouter/.env.example integrations/clawrouter/.env   # optional: tune ratio / threshold
docker compose -f integrations/clawrouter/docker-compose.yml up -d --build   # first build bakes the ~1.2 GB model in
./integrations/clawrouter/scripts/smoke.sh                                    # GET /health + a real /compress round-trip
```

`smoke.sh` should print something like:

```
==> GET http://localhost:8459/health
{"status":"ok","mode":"on"}
==> POST http://localhost:8459/compress  (long prose; expect output_tokens < input_tokens)
  method=lingua  in=4081  out=1744  ratio=0.427
  OK — saved 57.3% on prose
smoke: PASS
```

### GPU (production)

LLMLingua-2 is much faster on a GPU (≈ 46 ms vs. multi-second per call). Needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
docker compose -f integrations/clawrouter/docker-compose.yml --profile gpu up -d --build
```

> **torch / CUDA must match your GPU.** The GPU image defaults to CUDA 12.4 (Ada/Hopper).
> For Blackwell (RTX 50-series, e.g. **5090 / sm_120**) you need torch ≥ 2.7 on CUDA 12.8:
> ```bash
> docker build -f integrations/clawrouter/Dockerfile.gpu \
>   --build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 \
>   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
>   -t leanctx-layer8:gpu .
> ```

---

## The `/compress` contract

This is the interface the **P2b connector** codes against. The service is a pure function
over a message list — same shape in, same shape out, message count preserved.

**`POST /compress`**

```jsonc
// request
{
  "messages": [
    { "role": "system",    "content": "..." },
    { "role": "user",      "content": "<long prose document>" },
    { "role": "assistant", "content": "..." },
    { "role": "tool",      "content": "<tool result>" }   // forwarded VERBATIM
  ]
}
```

```jsonc
// response
{
  "messages": [ /* same list, eligible prose turns compressed in place */ ],
  "stats": {
    "input_tokens": 2304,
    "output_tokens": 1162,
    "ratio": 0.504,            // output/input — lower is more compression
    "method": "lingua",        // or "passthrough" when nothing was eligible
    "cost_usd": 0.0            // 0 for local LLMLingua-2
  },
  "compressed_message_count": 1
}
```

**Contract guarantees** (what a connector can rely on):

- **Message count is preserved** — one-in-one-out per message; splice back by index.
- **Only `system` / `user` / `assistant` turns with *string* `content` are eligible.**
  `tool` messages and any non-string (multimodal / content-list) message are returned
  **byte-for-byte unchanged**. A lossy prose pass must never touch tool results or images.
- **Short turns are protected** — turns under `LEANCTX_SERVER_THRESHOLD` (default 1500
  tokens) pass through uncompressed, so system prompts / terse instructions are safe.
- **Fail-open is the caller's job.** If the sidecar is down or errors, the connector should
  fall back to the un-compressed messages (ClawRouter already wraps Layer 8 in try/catch).

**`GET /health`** → `{"status":"ok","mode":"on"}` — for container/orchestrator probes.

---

## Configuration

All optional; the image ships with ClawRouter-tuned defaults. Override via `.env`,
compose `environment:`, or `docker run -e`.

| Env var | Default | Meaning |
|---|---|---|
| `LEANCTX_SERVER_MODE` | `on` | `on` compresses; `off` = passthrough smoke test (no model load) |
| `LEANCTX_SERVER_THRESHOLD` | `1500` | Don't compress turns below this many tokens |
| `LEANCTX_SERVER_LINGUA_RATIO` | `0.5` | Fraction of prose tokens to **keep** (lower = more aggressive) |
| `LEANCTX_SERVER_LINGUA_DEVICE` | `cpu` / `cuda` | Inference device (GPU image sets `cuda`) |
| `LEANCTX_SERVER_DEDUP` | `off` | Leave off — ClawRouter Layer 1 already deduplicates |
| `LEANCTX_SERVER_CONFIG` | — | Full JSON Middleware config; wins over all of the above |

The same sidecar is what the
[Phase-1 benchmark](../../benchmarks/clawrouter) already measures — this directory just
packages it for deployment.
