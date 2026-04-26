"""AC-7 / AC-8 / AC-10: bench CLI argument parsing + JSON output shape."""

from __future__ import annotations

import json
from typing import Any

from leanctx.bench import cli, scenarios
from leanctx.bench.schema import BenchRecord, validate_record


def test_list_human_output_includes_three_scenarios(capsys: Any) -> None:
    rc = cli.main(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    for name in ("lingua-local", "anthropic-e2e", "agent-structural"):
        assert name in captured.out


def test_list_json_output_is_valid(capsys: Any) -> None:
    rc = cli.main(["list", "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "scenarios" in payload
    names = {s["name"] for s in payload["scenarios"]}
    assert {"lingua-local", "anthropic-e2e", "agent-structural"}.issubset(names)


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
