"""Classifier tests — code / error / repeat / prose detection."""

from leanctx import ContentType, RepeatTracker, classify

# --------------------------------------------------------------------------- #
# Error detection
# --------------------------------------------------------------------------- #


def test_python_traceback_is_error() -> None:
    msg = {
        "role": "user",
        "content": (
            "Traceback (most recent call last):\n"
            '  File "x.py", line 3, in <module>\n'
            "ValueError: boom"
        ),
    }
    assert classify(msg) == ContentType.ERROR


def test_rust_panic_is_error() -> None:
    msg = {"role": "user", "content": "thread 'main' panicked at src/main.rs:12:5"}
    assert classify(msg) == ContentType.ERROR


def test_java_exception_is_error() -> None:
    msg = {
        "role": "user",
        "content": "Exception in thread \"main\" java.lang.NullPointerException",
    }
    assert classify(msg) == ContentType.ERROR


# --------------------------------------------------------------------------- #
# Code detection
# --------------------------------------------------------------------------- #


def test_fenced_code_block_is_code() -> None:
    msg = {
        "role": "user",
        "content": "Here is a snippet:\n```python\ndef foo():\n    pass\n```\nthoughts?",
    }
    assert classify(msg) == ContentType.CODE


def test_multi_line_def_imports_is_code() -> None:
    msg = {
        "role": "user",
        "content": "import os\nimport sys\ndef main():\n    pass",
    }
    assert classify(msg) == ContentType.CODE


def test_single_keyword_is_not_code() -> None:
    # Single "import" in prose shouldn't trip the classifier.
    msg = {"role": "user", "content": "The import statement in Python is tricky."}
    assert classify(msg) == ContentType.PROSE


# --------------------------------------------------------------------------- #
# Prose + edge cases
# --------------------------------------------------------------------------- #


def test_plain_prose_is_prose() -> None:
    msg = {"role": "user", "content": "Please explain this product in simple terms."}
    assert classify(msg) == ContentType.PROSE


# --------------------------------------------------------------------------- #
# Structured-data detection (route JSON to verbatim, never the lossy prose pass)
# --------------------------------------------------------------------------- #


def test_json_dialogue_blob_is_structured() -> None:
    # A serialized session/dialogue log — the LongBench shape LLMLingua-2 shreds.
    blob = (
        '{"meta": {"name": "bar_game", "rounds": 10}, "records": ['
        + ", ".join(
            f'{{"round_id": {i}, "responses": "go", "go_num": {i % 3}, '
            f'"winner": "stay", "utility": {i * 2}}}'
            for i in range(40)
        )
        + "]}"
    )
    assert len(blob) >= 200
    assert classify({"role": "user", "content": blob}) == ContentType.STRUCTURED


def test_long_prose_is_not_structured() -> None:
    # Well over the min length but no JSON keys → stays prose (compressible).
    prose = (
        "The quarterly review covered customer retention, onboarding friction, "
        "and the roadmap for the next two releases. Several teams raised concerns "
        "about latency under load, and we agreed to revisit the caching strategy "
        "before the launch. Nothing here is structured data; it is ordinary text."
    )
    assert len(prose) >= 200
    assert classify({"role": "user", "content": prose}) == ContentType.PROSE


def test_short_json_snippet_is_not_structured() -> None:
    # Below the min-length guard: too short to trust the density estimate.
    msg = {"role": "user", "content": '{"ok": true, "n": 3}'}
    assert classify(msg) != ContentType.STRUCTURED


def test_code_wins_over_structured() -> None:
    # A fenced JSON block is still CODE — both route to verbatim, but the CODE
    # check precedes STRUCTURED, so the established ordering is preserved.
    fenced = '```json\n' + '{"a": 1, "b": 2, "c": 3}\n' * 20 + '```'
    assert classify({"role": "user", "content": fenced}) == ContentType.CODE


def test_structured_routes_to_verbatim_by_default() -> None:
    # The default Router maps only prose→lingua; STRUCTURED is unmapped and so
    # falls back to the Verbatim default — content is never corrupted.
    from leanctx.compressors import Verbatim
    from leanctx.router import Router

    router = Router(default=Verbatim())
    assert router.route(ContentType.STRUCTURED).name == Verbatim().name


def test_empty_content_is_unknown() -> None:
    assert classify({"role": "user", "content": ""}) == ContentType.UNKNOWN


def test_whitespace_only_is_unknown() -> None:
    assert classify({"role": "user", "content": "   \n\t  "}) == ContentType.UNKNOWN


def test_missing_content_key_is_unknown() -> None:
    assert classify({"role": "user"}) == ContentType.UNKNOWN


def test_anthropic_content_blocks_are_classified() -> None:
    # Multiple text blocks, one of which is fenced code.
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Check this:"},
            {"type": "text", "text": "```\nls -la\n```"},
        ],
    }
    assert classify(msg) == ContentType.CODE


def test_error_wins_over_code_when_both_present() -> None:
    # If a stack trace also contains code-looking lines, it's still an error.
    msg = {
        "role": "user",
        "content": (
            "Traceback (most recent call last):\n"
            "  File x.py line 3\n"
            "def foo():\n"
            "    pass"
        ),
    }
    assert classify(msg) == ContentType.ERROR


# --------------------------------------------------------------------------- #
# Repeat tracking
# --------------------------------------------------------------------------- #


def test_repeat_first_occurrence_is_not_a_repeat() -> None:
    t = RepeatTracker()
    assert t.is_repeat({"role": "user", "content": "hello"}) is False


def test_repeat_second_occurrence_is_a_repeat() -> None:
    t = RepeatTracker()
    msg = {"role": "user", "content": "duplicate tool output"}
    t.is_repeat(msg)
    assert t.is_repeat(msg) is True


def test_repeat_different_content_is_not_confused() -> None:
    t = RepeatTracker()
    t.is_repeat({"role": "user", "content": "first"})
    assert t.is_repeat({"role": "user", "content": "second"}) is False


def test_repeat_empty_message_never_flags() -> None:
    t = RepeatTracker()
    assert t.is_repeat({"role": "user", "content": ""}) is False
    assert t.is_repeat({"role": "user", "content": ""}) is False


def test_repeat_reset_clears_state() -> None:
    t = RepeatTracker()
    msg = {"role": "user", "content": "hello"}
    t.is_repeat(msg)
    t.reset()
    assert t.is_repeat(msg) is False
