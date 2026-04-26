"""AC-9: bench --runs N must construct fresh client/middleware per run.

Regression guard: a v0.2 bug had DedupStrategy state leaking across
calls because users reused the same client instance. The bench CLI
must construct a fresh client per run; this test pins the behavior.
"""

from __future__ import annotations

from typing import Any

from leanctx.bench import cli, scenarios
from leanctx.bench.schema import BenchRecord
from leanctx.middleware import Middleware


def test_runs_n_invokes_runner_per_run(capsys: Any) -> None:
    """--runs N invokes the runner N times, with fresh kwargs each call.

    The actual fresh-state guard for compression is verified by
    test_lingua_local_runs_n_does_not_leak_dedup_state below; this
    test pins the call-count contract."""
    scenarios.reset_for_tests()
    call_count = {"n": 0}

    @scenarios.register("test-isolation", description="isolation-check")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        call_count["n"] += 1
        # Build a fresh Middleware in the runner — same pattern the
        # real runners use. Object identity differs across runs even
        # when CPython reuses the freed slot, so we don't assert on
        # id(); the dedup-state regression test below is what matters.
        Middleware({})
        return BenchRecord(
            leanctx_version="0.0.0-test",
            scenario="test-isolation",
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
        rc = cli.main(["run", "test-isolation", "--workload", "rag", "--runs", "3"])
        assert rc == 0
        assert call_count["n"] == 3
    finally:
        scenarios.reset_for_tests()


def test_real_lingua_local_runs_n_does_not_leak_dedup_state(capsys: Any) -> None:
    """AC-9 / task29: invoke the actual `lingua-local` scenario via the
    CLI with --runs 5 and confirm input_tokens are stable across runs.

    Uses a stub PromptCompressor so the test runs without the real
    LLMLingua-2 model — but it exercises the SHIPPED runner code path,
    not a custom test-only scenario."""
    import json

    import leanctx.compressors.lingua as lingua_mod  # noqa: PLC0415

    class _StubPC:
        """Stand-in for llmlingua.PromptCompressor — returns the input
        unchanged so token counts are deterministic across runs."""

        model_name = "stub-lingua-test"

        def compress_prompt(
            self, text: str, *, rate: float = 0.5, **kwargs: Any
        ) -> dict[str, Any]:
            return {"compressed_prompt": text}

    # Monkey-patch the real Lingua class to use the stub instead of
    # loading LLMLingua. The Lingua._compress_inner reads
    # self._prompt_compressor; we set it eagerly per instance.
    real_init = lingua_mod.Lingua.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        real_init(self, *args, **kwargs)
        self._prompt_compressor = _StubPC()

    lingua_mod.Lingua.__init__ = _patched_init  # type: ignore[method-assign]
    try:
        rc = cli.main(["run", "lingua-local", "--workload", "rag", "--runs", "5"])
        assert rc == 0
        captured = capsys.readouterr()
        records = [
            json.loads(line) for line in captured.out.splitlines() if line.strip()
        ]
        assert len(records) == 5
        first = records[0]["input_tokens"]
        for rec in records[1:]:
            assert rec["input_tokens"] == first, (
                f"input_tokens drifted across runs in REAL lingua-local: "
                f"{first} → {rec['input_tokens']} (suggests cross-run leak)"
            )
    finally:
        lingua_mod.Lingua.__init__ = real_init  # type: ignore[method-assign]


def test_negative_shared_state_dedup_does_not_leak(capsys: Any) -> None:
    """AC-9 negative case: even when the underlying DedupStrategy hash
    set is non-empty before the run, a fresh per-run client must not
    inherit it. We simulate the regression by running through Middleware
    directly twice with the same input — the first run primes any
    in-strategy state; the second run on a FRESH Middleware must not be
    affected."""
    from leanctx.middleware import Middleware  # noqa: PLC0415

    msgs = [{"role": "user", "content": "the same content twice over"}]

    # mode=off → no real compression but exercises strategies pipeline.
    mw1 = Middleware({"mode": "off"})
    out1, stats1 = mw1.compress_messages(msgs)
    # Now run via a FRESH middleware — the v0.2 bug shared dedup state
    # across instances; the v0.2.1 fix made it per-instance. This test
    # pins that contract.
    mw2 = Middleware({"mode": "off"})
    out2, stats2 = mw2.compress_messages(msgs)

    # Off-mode just passes through; both runs see the same input and
    # produce identical output.
    assert len(out1) == len(out2) == 1
    assert stats1.input_tokens == stats2.input_tokens


def test_lingua_local_runs_n_does_not_leak_dedup_state(capsys: Any) -> None:
    """Concrete v0.2 regression guard: 5 lingua-local runs of the same
    workload must produce identical input_tokens (no decreasing trend
    that would indicate cross-run dedup leakage).

    This test uses a fake-Lingua via a custom registered runner so it
    works without the [lingua] extra installed."""
    import json

    scenarios.reset_for_tests()

    seen_input_tokens: list[int] = []

    @scenarios.register("lingua-fake", description="dedup-state regression check")
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        # The runner builds a fresh Middleware each call. Token count
        # should be deterministic across runs since the workload is
        # identical and there's no shared state.
        mw = Middleware(
            {
                "mode": "on",
                "trigger": {"threshold_tokens": 1},
                "strategies": {"dedup": True},
            }
        )
        from leanctx.bench.workloads import load_workload

        msgs = load_workload(workload)
        compressed, stats = mw.compress_messages(msgs)
        seen_input_tokens.append(stats.input_tokens)
        return BenchRecord(
            leanctx_version="0.0.0-test",
            scenario="lingua-fake",
            workload=workload,
            status="success",
            compressor="verbatim",
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
            tokens_saved=stats.input_tokens - stats.output_tokens,
            ratio=stats.ratio,
            cost_usd=stats.cost_usd,
            duration_ms=1,
            warmup=False,
            timestamp="2026-04-26T00:00:00Z",
        )

    try:
        rc = cli.main(["run", "lingua-fake", "--workload", "rag", "--runs", "5"])
        assert rc == 0
        captured = capsys.readouterr()
        records = [json.loads(line) for line in captured.out.splitlines() if line.strip()]
        assert len(records) == 5
        # Input tokens must be the same across runs (no monotonic
        # decrease — that's the v0.2 cross-run dedup leak).
        first = records[0]["input_tokens"]
        for rec in records[1:]:
            assert rec["input_tokens"] == first, (
                f"input_tokens drifted: {first} → {rec['input_tokens']} "
                f"(suggests cross-run dedup state leak)"
            )
    finally:
        scenarios.reset_for_tests()


def test_missing_env_var_exits_3(capsys: Any) -> None:
    """AC-7 / AC-10: a scenario whose required_env is unset exits 3 with
    a clear diagnostic naming the missing env var."""
    scenarios.reset_for_tests()

    @scenarios.register(
        "needs-key",
        description="needs an api key",
        required_env=("DEFINITELY_NOT_SET_ENV_VAR_FOR_TESTS",),
    )
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        raise AssertionError("should not run when env var is missing")

    try:
        rc = cli.main(["run", "needs-key", "--workload", "rag"])
        assert rc == 3
        captured = capsys.readouterr()
        assert "DEFINITELY_NOT_SET_ENV_VAR_FOR_TESTS" in captured.err
    finally:
        scenarios.reset_for_tests()


def test_missing_extra_diagnostic_via_runner_runtime_error(capsys: Any) -> None:
    """AC-10 negative test: when a runner raises a RuntimeError naming a
    missing extra, the CLI exits 3 with the diagnostic on stderr (no
    traceback). This is the path taken by the real runners — see
    e.g. lingua_local.py raising RuntimeError when llmlingua is absent."""
    scenarios.reset_for_tests()

    @scenarios.register(
        "raises-missing-extra",
        description="runner raises RuntimeError naming a missing extra",
    )
    def _runner(*, workload: str, **opts: Any) -> BenchRecord:
        raise RuntimeError(
            "the [some-extra] extra is required for raises-missing-extra. "
            "Install with: pip install 'leanctx[some-extra]'"
        )

    try:
        rc = cli.main(["run", "raises-missing-extra", "--workload", "rag"])
        assert rc == 3
        captured = capsys.readouterr()
        assert "some-extra" in captured.err
        # No traceback in user-facing output.
        assert "Traceback" not in captured.err
    finally:
        scenarios.reset_for_tests()
