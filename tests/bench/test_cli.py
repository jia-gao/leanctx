"""AC-7 / AC-8 / AC-10: bench CLI argument parsing + JSON output shape."""

from __future__ import annotations

import json
from typing import Any

from leanctx.bench import cli, scenarios
from leanctx.bench.schema import BenchRecord, validate_record

_REQUIRED_SCENARIOS = {
    "lingua-local",
    "anthropic-e2e",
    "selfllm-anthropic",
    "selfllm-openai",
    "selfllm-gemini",
    "agent-structural",
}


def test_list_human_output_includes_all_six_scenarios(capsys: Any) -> None:
    """AC-7: bench list must show all six plan-required scenarios."""
    rc = cli.main(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    for name in _REQUIRED_SCENARIOS:
        assert name in captured.out, f"missing scenario {name!r} in `bench list` output"


def test_bench_prefix_is_optional(capsys: Any) -> None:
    """The entry point is `leanctx`, but README/docs say `leanctx bench list/run`.
    The CLI must accept BOTH `leanctx list` and `leanctx bench list`."""
    rc = cli.main(["bench", "list"])
    assert rc == 0
    captured = capsys.readouterr()
    for name in _REQUIRED_SCENARIOS:
        assert name in captured.out


def test_list_json_output_is_valid(capsys: Any) -> None:
    rc = cli.main(["list", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "scenarios" in payload
    names = {s["name"] for s in payload["scenarios"]}
    assert _REQUIRED_SCENARIOS.issubset(names), (
        f"missing required scenarios: {_REQUIRED_SCENARIOS - names}"
    )


def test_reset_for_tests_restores_builtins(capsys: Any) -> None:
    """Regression guard: after reset_for_tests(), the next list_scenarios()
    call must re-register the built-in scenarios. Previous bug: cleared
    _LOADED + _REGISTRY, but Python module cache prevented re-registration."""
    scenarios.reset_for_tests()
    # First public call re-loads the built-ins.
    rc = cli.main(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    for name in _REQUIRED_SCENARIOS:
        assert name in captured.out, (
            f"reset_for_tests() did not restore built-in scenario {name!r}"
        )


def test_unknown_scenario_exits_with_2(capsys: Any) -> None:
    rc = cli.main(["run", "does-not-exist", "--workload", "rag"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown scenario" in captured.err


def test_unknown_workload_exits_with_2(capsys: Any) -> None:
    rc = cli.main(["run", "lingua-local", "--workload", "nonexistent"])
    assert rc == 2
    captured = capsys.readouterr()
    assert "unknown workload" in captured.err


def test_run_with_fake_runner_emits_valid_record(capsys: Any) -> None:
    """Inject a fake runner so we don't need lingua/respx to test CLI flow."""
    scenarios.reset_for_tests()

    @scenarios.register("test-fake", description="fake")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        return BenchRecord(
            leanctx_version="0.0.0-test",
            scenario="test-fake",
            workload=workload,
            status="success",
            compressor="verbatim",
            input_tokens=10,
            output_tokens=10,
            tokens_saved=0,
            ratio=1.0,
            cost_usd=0.0,
            duration_ms=1,
            warmup=False,
            timestamp="2026-04-26T00:00:00Z",
        )

    try:
        rc = cli.main(["run", "test-fake", "--workload", "rag"])
        assert rc == 0
        captured = capsys.readouterr()
        record = json.loads(captured.out.strip())
        errors = validate_record(record)
        assert errors == [], f"schema validation produced errors: {errors}"
        assert record["scenario"] == "test-fake"
        assert record["workload"] == "rag"
        assert record["schema_version"] == "1"
    finally:
        scenarios.reset_for_tests()


def test_run_runs_n_emits_n_records(capsys: Any) -> None:
    scenarios.reset_for_tests()

    @scenarios.register("test-fake", description="fake")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        return BenchRecord(
            leanctx_version="0.0.0-test",
            scenario="test-fake",
            workload=workload,
            status="success",
            compressor="verbatim",
            input_tokens=1,
            output_tokens=1,
            tokens_saved=0,
            ratio=1.0,
            cost_usd=0.0,
            duration_ms=1,
            warmup=False,
            timestamp="2026-04-26T00:00:00Z",
        )

    try:
        rc = cli.main(["run", "test-fake", "--workload", "rag", "--runs", "3"])
        assert rc == 0
        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) == 3
        for line in lines:
            assert validate_record(json.loads(line)) == []
    finally:
        scenarios.reset_for_tests()


def test_run_failure_exits_nonzero(capsys: Any) -> None:
    scenarios.reset_for_tests()

    @scenarios.register("test-failing", description="fails")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        return BenchRecord(
            leanctx_version="0.0.0-test",
            scenario="test-failing",
            workload=workload,
            status="failure",
            compressor="lingua",
            input_tokens=1,
            output_tokens=1,
            tokens_saved=0,
            ratio=1.0,
            cost_usd=0.0,
            duration_ms=1,
            warmup=False,
            timestamp="2026-04-26T00:00:00Z",
            error="invariant X failed",
        )

    try:
        rc = cli.main(["run", "test-failing", "--workload", "agent"])
        assert rc == 1
    finally:
        scenarios.reset_for_tests()


def test_run_runtime_error_exits_3(capsys: Any) -> None:
    scenarios.reset_for_tests()

    @scenarios.register("test-needs-extra", description="missing extra")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        raise RuntimeError("the [lingua] extra is required for ...")

    try:
        rc = cli.main(["run", "test-needs-extra", "--workload", "rag"])
        assert rc == 3
        captured = capsys.readouterr()
        assert "lingua" in captured.err
    finally:
        scenarios.reset_for_tests()


def test_required_extras_cli_preflight_exits_3_before_runner(
    capsys: Any, monkeypatch: Any
) -> None:
    """AC-10: deterministic no-extras diagnostic. The CLI's preflight in
    `_cmd_run` must reject missing required_extras BEFORE the runner is
    invoked. Pin the contract: runner is never called, exit 3, scenario
    name + extra name on stderr.

    This is the path distinct from a runner-thrown RuntimeError — the
    preflight in `cli._cmd_run` calls `_missing_extras(required_extras)`
    which probes via `_extra_installed`. We monkeypatch that probe to
    force-fail for our synthetic extra so this test is deterministic
    regardless of which extras the test environment has installed."""
    scenarios.reset_for_tests()

    runner_called = {"n": 0}

    @scenarios.register(
        "test-preflight-extras",
        description="required_extras preflight check",
        required_extras=("synthetic-test-extra",),
    )
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        runner_called["n"] += 1
        raise AssertionError("runner must not be invoked when required_extra is missing")

    real_extra_installed = cli._extra_installed

    def _patched(name: str) -> bool:
        if name == "synthetic-test-extra":
            return False
        return real_extra_installed(name)

    monkeypatch.setattr(cli, "_extra_installed", _patched)

    try:
        rc = cli.main(["run", "test-preflight-extras", "--workload", "rag"])
        assert rc == 3
        assert runner_called["n"] == 0, (
            "CLI preflight failed to gate; runner was invoked despite missing extra"
        )
        captured = capsys.readouterr()
        assert "test-preflight-extras" in captured.err
        assert "synthetic-test-extra" in captured.err
        assert "Traceback" not in captured.err
    finally:
        scenarios.reset_for_tests()


def test_record_validation_rejects_missing_required_field() -> None:
    incomplete = {"schema_version": "1", "scenario": "x"}
    errors = validate_record(incomplete)
    assert any("missing required field" in e for e in errors)


def test_record_validation_rejects_schema_version_mismatch() -> None:
    record = BenchRecord(
        leanctx_version="0.0.0",
        scenario="x",
        workload="rag",
        timestamp="2026-04-26T00:00:00Z",
    ).to_dict()
    record["schema_version"] = "999"
    errors = validate_record(record)
    assert any("schema_version mismatch" in e for e in errors)
