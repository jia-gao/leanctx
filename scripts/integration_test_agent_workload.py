"""Integration test on a realistic coding-agent transcript.

This is leanctx's strongest case: agent traffic with mixed content
where prose / tool_result text gets compressed, code blocks and
errors get preserved verbatim, and tool_use → tool_result linkage
survives the pipeline.

Builds a representative ~6 KB transcript (user prompt → file read →
grep → failed edit → retry → success → final answer) and runs it
through ``leanctx.Anthropic`` with ``mode="on"`` and the full
content-aware routing config. Reports:

* Total input/output tokens (the headline number)
* Per-message before/after sizes so you can see what compressed
  and what didn't
* Verification that tool_use_id linkage survived
* Verification that code blocks were preserved verbatim

Requires: ``pip install 'leanctx[lingua,anthropic,bench]'``. Runs
against the real LLMLingua-2 model (1.2 GB; downloads on first run,
cached afterwards) with respx mocking the Anthropic endpoint so no
API key is needed.

Run with::

    .venv/bin/python scripts/integration_test_agent_workload.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import respx
from httpx import Response

from leanctx import Anthropic
from leanctx._content import get_text_content
from leanctx.tokens import count_message_tokens

# --------------------------------------------------------------------------- #
# Realistic agent transcript fixture
# --------------------------------------------------------------------------- #

# Python source — this is exactly the kind of code block that MUST be
# preserved verbatim. A compressor that drops a closing brace or rewrites
# an `if` statement would break tool linkage and silently corrupt code.
_AUTH_MIDDLEWARE_PY = '''\
"""JWT auth middleware for the user-facing API.

Verifies the bearer token, loads the user from the cache or DB, and
attaches the user to the request scope.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import jwt
from fastapi import HTTPException, Request, status

from app.cache import user_cache
from app.config import settings
from app.db import get_user_by_id
from app.observability import metrics

logger = logging.getLogger(__name__)


async def authenticate(request: Request) -> dict[str, Any]:
    """Verify the bearer token and load the user.

    Raises HTTPException(401) on missing/invalid tokens.
    Caches the user object for cache_ttl_seconds after first lookup.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or malformed Authorization header",
        )
    token = auth_header[len("Bearer "):]

    try:
        # NB: this is the line that broke after the migration — the
        # original code passed algorithms=["HS256"] but the new helper
        # forgot to forward it.
        payload = jwt.decode(token, settings.jwt_secret)
    except jwt.PyJWTError as e:
        metrics.auth_failures.inc()
        logger.warning("jwt decode failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"invalid token: {e}",
        ) from e

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token missing sub claim",
        )

    cached = await user_cache.get(user_id)
    if cached is not None:
        return cached

    user = await get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user not found",
        )
    await user_cache.set(user_id, user, ttl=settings.cache_ttl_seconds)
    return user
'''

_GREP_OUTPUT = """\
src/auth/middleware.py:31:        payload = jwt.decode(token, settings.jwt_secret)
src/auth/legacy_middleware.py:88:        payload = jwt.decode(token, secret, algorithms=["HS256"])
src/auth/test_middleware.py:14:        payload = jwt.decode(self.fake_token, "secret", algorithms=["HS256"])
src/auth/test_middleware.py:42:        payload = jwt.decode(self.fake_token, "secret", algorithms=["HS256"])
tests/integration/test_auth_flow.py:201:    payload = jwt.decode(token, secret, algorithms=["HS256"])
tests/integration/test_auth_flow.py:418:    payload = jwt.decode(token, secret, algorithms=["HS256"])
tools/migrate_auth.py:64:    payload = jwt.decode(token, secret, algorithms=["HS256"])
"""

_LOG_OUTPUT = """\
2026-04-25T14:18:02.314Z INFO  auth-service Starting up version 2026.4.18
2026-04-25T14:18:02.401Z INFO  auth-service Connected to user_db (pool size 32)
2026-04-25T14:18:14.812Z DEBUG auth-service request id=abc-001 path=/me method=GET
2026-04-25T14:18:14.815Z WARN  auth-service jwt decode failed: It is required that you pass in a value for the "algorithms" argument when calling decode().
2026-04-25T14:18:14.815Z INFO  auth-service request id=abc-001 status=401 latency_ms=3
2026-04-25T14:18:14.892Z DEBUG auth-service request id=abc-002 path=/orders method=GET
2026-04-25T14:18:14.895Z WARN  auth-service jwt decode failed: It is required that you pass in a value for the "algorithms" argument when calling decode().
2026-04-25T14:18:14.895Z INFO  auth-service request id=abc-002 status=401 latency_ms=3
2026-04-25T14:18:15.011Z DEBUG auth-service request id=abc-003 path=/orders method=POST
2026-04-25T14:18:15.014Z WARN  auth-service jwt decode failed: It is required that you pass in a value for the "algorithms" argument when calling decode().
2026-04-25T14:18:15.014Z INFO  auth-service request id=abc-003 status=401 latency_ms=3
""" * 3  # ~30 lines of repetitive log output, prime compression target

_EDIT_FAILURE = """\
Traceback (most recent call last):
  File "/app/tools/edit.py", line 88, in apply_edit
    raise FileChangedError(f"file modified since last read at {since}")
app.tools.errors.FileChangedError: file modified since last read at 2026-04-25T14:30:11Z

The file src/auth/middleware.py was modified by another process. Re-read it
before applying further edits.
"""


def _build_transcript() -> list[dict[str, Any]]:
    """Build a 7-message coding-agent transcript with mixed content."""
    return [
        # 1. User asks for help debugging
        {
            "role": "user",
            "content": (
                "Our production auth service is rejecting valid JWTs after the "
                "recent migration to a shared jwt-helper module. The CI passes "
                "but live traffic 401s on every request. Can you find the bug?"
            ),
        },
        # 2. Assistant: reasoning + read_file tool call
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I'll start by reading the current auth middleware to "
                        "understand the JWT verification flow, then check git "
                        "history and run a grep to compare against legacy code "
                        "that's known to work."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01_read_middleware",
                    "name": "read_file",
                    "input": {"path": "src/auth/middleware.py"},
                },
            ],
        },
        # 3. tool_result: full Python source — code, must be preserved verbatim
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01_read_middleware",
                    "content": _AUTH_MIDDLEWARE_PY,
                }
            ],
        },
        # 4. Assistant: reasoning + grep
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I see the suspicious line — jwt.decode is called without "
                        "the algorithms kwarg, which PyJWT requires since 2.0 to "
                        "prevent algorithm-confusion attacks. Let me grep across "
                        "the codebase to confirm the pattern in legacy code."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": "toolu_02_grep",
                    "name": "grep",
                    "input": {
                        "pattern": "jwt\\.decode\\(",
                        "path": "src/auth tests/ tools/",
                        "output_mode": "content",
                    },
                },
            ],
        },
        # 5. tool_result: grep output — code-like, classifier should keep verbatim
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_02_grep",
                    "content": _GREP_OUTPUT,
                }
            ],
        },
        # 6. Assistant: long verbose log dump request
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Confirmed: every other call site passes the algorithms "
                        "kwarg explicitly. The new middleware is the only one "
                        "missing it. Let me also pull the recent error logs so I "
                        "can quote them in the fix's commit message."
                    ),
                },
                {
                    "type": "tool_use",
                    "id": "toolu_03_logs",
                    "name": "run_command",
                    "input": {"command": "kubectl logs deploy/auth -n prod --since=1h"},
                },
            ],
        },
        # 7. tool_result: log dump — prose-like, prime compression target
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_03_logs",
                    "content": _LOG_OUTPUT,
                }
            ],
        },
        # 8. Assistant: try to edit, fails
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Now I'll patch middleware.py to pass algorithms=['HS256'] explicitly.",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_04_edit",
                    "name": "edit_file",
                    "input": {
                        "path": "src/auth/middleware.py",
                        "old_str": "payload = jwt.decode(token, settings.jwt_secret)",
                        "new_str": (
                            "payload = jwt.decode(token, settings.jwt_secret, "
                            "algorithms=[settings.jwt_algorithm])"
                        ),
                    },
                },
            ],
        },
        # 9. tool_result: error — must be preserved verbatim (classifier:ERROR -> Verbatim)
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_04_edit",
                    "content": _EDIT_FAILURE,
                    "is_error": True,
                }
            ],
        },
    ]


# --------------------------------------------------------------------------- #
# Tool-linkage verification
# --------------------------------------------------------------------------- #


def _collect_tool_use_ids(messages: list[dict[str, Any]]) -> list[str]:
    """Collect the tool_use ids that appear in tool_use blocks AND the
    tool_use_ids that appear in tool_result blocks. Both should still
    pair up after compression."""
    use_ids: list[str] = []
    result_ids: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                use_ids.append(block.get("id", ""))
            elif btype == "tool_result":
                result_ids.append(block.get("tool_use_id", ""))
    return use_ids + result_ids


def _find_block(
    messages: list[dict[str, Any]], block_type: str, attr: str, value: str
) -> dict[str, Any] | None:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == block_type
                and block.get(attr) == value
            ):
                return block
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _mock_anthropic_response() -> dict[str, Any]:
    return {
        "id": "msg_01TEST_AGENT",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "ack"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 2,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def main() -> None:
    transcript = _build_transcript()

    print("=== Coding-agent workload integration test ===\n")
    print(f"Transcript: {len(transcript)} messages")
    total_chars = sum(len(get_text_content(m)) for m in transcript)
    total_tokens = count_message_tokens(transcript)
    print(f"Original size: {total_chars} chars, ~{total_tokens} tokens\n")

    # leanctx config — the realistic production setting
    leanctx_config: dict[str, Any] = {
        "mode": "on",
        "trigger": {"threshold_tokens": 0},  # always run (small transcript)
        "routing": {
            "code": "verbatim",
            "error": "verbatim",
            "prose": "lingua",
            "long_important": "lingua",
        },
        "lingua": {"ratio": 0.5},
        # Strategies on by default in active mode
    }

    print("Config:")
    print(f"  mode            = {leanctx_config['mode']}")
    print(f"  routing         = {leanctx_config['routing']}")
    print(f"  lingua.ratio    = {leanctx_config['lingua']['ratio']}\n")

    # End-to-end: leanctx.Anthropic + real Lingua model + respx-mocked Anthropic
    print("Sending through leanctx.Anthropic (real Lingua, mocked HTTP)...")
    print("(First run downloads ~1.2 GB of LLMLingua-2 weights; cached afterwards.)")
    t0 = time.monotonic()
    # assert_all_called=False because the HuggingFace pass-through
    # routes are only hit on first run / template-freshness checks; on
    # cached runs they're unused and respx would otherwise flag that.
    with respx.mock(
        base_url="https://api.anthropic.com", assert_all_called=False
    ) as mock:
        route = mock.post("/v1/messages").mock(
            return_value=Response(200, json=_mock_anthropic_response())
        )
        # Let HuggingFace requests through to the real network so the
        # LLMLingua-2 model can download (first run) or do its cache /
        # template freshness checks (subsequent runs). Without this,
        # respx's "all unmocked requests fail" rule blocks model load.
        mock.route(host__regex=r".*huggingface\.co").pass_through()
        mock.route(host__regex=r".*hf\.co").pass_through()

        client = Anthropic(api_key="sk-test", leanctx_config=leanctx_config)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=transcript,
        )
    elapsed = time.monotonic() - t0
    print(f"  full request cycle: {elapsed:.2f}s\n")

    # What did we actually send to api.anthropic.com?
    sent_body = json.loads(route.calls[0].request.content)
    sent_messages: list[dict[str, Any]] = sent_body["messages"]
    sent_chars = sum(len(get_text_content(m)) for m in sent_messages)
    sent_tokens = count_message_tokens(sent_messages)

    print("=== Results ===")
    print(f"  Original:    {total_chars:>6} chars, ~{total_tokens:>5} tokens")
    print(f"  Sent on wire:{sent_chars:>6} chars, ~{sent_tokens:>5} tokens")
    print(
        f"  Reduction:   {1 - sent_chars / total_chars:.1%} chars, "
        f"{1 - sent_tokens / total_tokens:.1%} tokens"
    )
    print(f"  leanctx_method:       {response.usage.leanctx_method}")
    print(f"  leanctx_ratio:        {response.usage.leanctx_ratio:.1%}")
    print(f"  leanctx_tokens_saved: {response.usage.leanctx_tokens_saved}\n")

    # --------------------------------------------------------------------- #
    # Per-message breakdown
    # --------------------------------------------------------------------- #
    print("=== Per-message before / after (chars) ===")
    print(f"  {'#':>2}  {'role':<10} {'before':>7} {'after':>7} {'Δ':>7}  {'kind'}")
    for i, (orig, sent) in enumerate(zip(transcript, sent_messages, strict=True)):
        before = len(get_text_content(orig))
        after = len(get_text_content(sent))
        delta = before - after
        kind = _describe_kind(orig)
        print(
            f"  {i:>2}  {orig.get('role', '?'):<10} "
            f"{before:>7} {after:>7} {delta:>7}  {kind}"
        )
    print()

    # --------------------------------------------------------------------- #
    # Verifications
    # --------------------------------------------------------------------- #
    print("=== Correctness checks ===")

    # 1. Tool linkage — every tool_use_id in tool_use blocks should still
    #    appear in a tool_result block, and vice versa.
    orig_ids = _collect_tool_use_ids(transcript)
    sent_ids = _collect_tool_use_ids(sent_messages)
    linkage_ok = orig_ids == sent_ids
    print(
        f"  tool_use_id linkage preserved: "
        f"{'✅' if linkage_ok else '❌'}  ({len(orig_ids)} ids)"
    )
    assert linkage_ok, f"tool ids changed: orig={orig_ids} sent={sent_ids}"

    # 2. Code blocks preserved verbatim. The middleware.py file is in a
    #    tool_result block; classifier should route it to Verbatim.
    src_block = _find_block(
        sent_messages, "tool_result", "tool_use_id", "toolu_01_read_middleware"
    )
    assert src_block is not None
    code_preserved = src_block["content"] == _AUTH_MIDDLEWARE_PY
    print(
        f"  code block preserved verbatim: "
        f"{'✅' if code_preserved else '❌'}  (auth/middleware.py)"
    )
    assert code_preserved, (
        "auth/middleware.py was mutated by the compressor — this would "
        "silently corrupt code in production. The classifier should route "
        "code blocks to Verbatim."
    )

    # 3. Error block preserved verbatim
    err_block = _find_block(sent_messages, "tool_result", "tool_use_id", "toolu_04_edit")
    assert err_block is not None
    err_preserved = err_block["content"] == _EDIT_FAILURE
    print(
        f"  error block preserved verbatim: "
        f"{'✅' if err_preserved else '❌'}  (file-modified error)"
    )
    assert err_preserved, (
        "Error tool_result was mutated — would strip diagnostic information "
        "agents rely on. Errors should route to Verbatim."
    )

    # 4. tool_use input fields preserved exactly (compressing these would
    #    change the actual tool invocation)
    edit_block = _find_block(sent_messages, "tool_use", "id", "toolu_04_edit")
    assert edit_block is not None
    grep_block = _find_block(sent_messages, "tool_use", "id", "toolu_02_grep")
    assert grep_block is not None
    inputs_preserved = (
        edit_block["input"]["new_str"]
        == "payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])"
    ) and (grep_block["input"]["pattern"] == "jwt\\.decode\\(")
    print(
        f"  tool_use input fields preserved: "
        f"{'✅' if inputs_preserved else '❌'}  (edit/grep inputs)"
    )
    assert inputs_preserved, "tool_use.input was mutated — would change tool semantics"

    # 5. The compressible target — log output — should have actually
    #    shrunk. If it didn't, classifier or router is misconfigured.
    log_block = _find_block(sent_messages, "tool_result", "tool_use_id", "toolu_03_logs")
    assert log_block is not None
    log_compressed = isinstance(log_block["content"], str) and len(
        log_block["content"]
    ) < len(_LOG_OUTPUT)
    log_after = (
        len(log_block["content"]) if isinstance(log_block["content"], str) else "n/a"
    )
    print(
        f"  log output was compressed:     "
        f"{'✅' if log_compressed else '❌'}  ({len(_LOG_OUTPUT)} → {log_after} chars)"
    )
    assert log_compressed, (
        f"log output didn't shrink ({len(_LOG_OUTPUT)} → {log_after}) — "
        "classifier or router likely misconfigured for the prose-heavy path."
    )

    print("\nOK")


def _describe_kind(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return "text"
    if isinstance(content, list):
        types = [b.get("type") for b in content if isinstance(b, dict)]
        return ", ".join(t for t in types if t)
    return "?"


if __name__ == "__main__":
    main()
