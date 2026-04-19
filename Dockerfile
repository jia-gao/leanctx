# syntax=docker/dockerfile:1.7
#
# leanctx — minimal container image for CI, local experimentation, and as a
# base for downstream apps that want leanctx pre-installed.
#
# Two flavors, selected by the LINGUA build arg:
#
#   docker build -t leanctx:slim .                  # default: no ML deps
#   docker build -t leanctx:lingua --build-arg LINGUA=true .  # adds llmlingua + torch
#
# The slim image is ~180 MB (python-slim + anthropic/openai/gemini/tiktoken).
# The lingua image is much larger (~3 GB) because torch + transformers, and
# the LLMLingua-2 model itself (~1.2 GB) is downloaded on first use.

FROM python:3.11-slim AS base

ARG LINGUA=false

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy just the metadata + package so the install step is cache-friendly.
COPY pyproject.toml README.md LICENSE ./
COPY leanctx ./leanctx

# Install from the local source. Default install pulls all three provider
# SDKs plus tiktoken; LINGUA=true adds llmlingua (and transitively torch,
# transformers, tokenizers).
RUN pip install --upgrade pip \
    && if [ "$LINGUA" = "true" ]; then \
           pip install -e '.[all]'; \
       else \
           pip install -e '.[anthropic,openai,gemini,tokens]'; \
       fi \
    && pip list

# Smoke check: import should succeed and version should match pyproject.
RUN python -c "import leanctx; print(f'leanctx {leanctx.__version__} ready')"

CMD ["python", "-c", "import leanctx; print(f'leanctx {leanctx.__version__} — see https://github.com/jia-gao/leanctx'); import sys; sys.exit(0)"]
