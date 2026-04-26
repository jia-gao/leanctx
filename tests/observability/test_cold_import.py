"""AC-11 — `import leanctx` cold-import budget.

Cold-import time for ``import leanctx`` must remain ≤ 80 ms in the
no-extras configuration. The budget is generous on purpose; the goal is
a regression guard, not a microbenchmark. v0.2 baseline measured ~50 ms
locally; v0.3 added the observability and bench packages (~+10–15 ms);
CI Linux runners add another ~5–10 ms over Apple Silicon. 80 ms gives
real headroom while still catching regressions that bloat startup.

The test is skipped on slow CI runners (recognized via ``CI`` +
``SLOW_RUNNER``) where cold-import variance can blow the budget for
non-leanctx reasons.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_BUDGET_MS = 80.0
_RUNS = 3


def test_cold_import_under_budget() -> None:
    if os.environ.get("CI") and os.environ.get("SLOW_RUNNER"):
        return  # explicit slow-runner opt-out

    repo_root = Path(__file__).resolve().parents[2]
    timings: list[float] = []
    for _ in range(_RUNS):
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-c", "import leanctx"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings.append(elapsed_ms)
        assert proc.returncode == 0
    best = min(timings)
    interpreter_overhead_ms = _measure_python_overhead(repo_root)
    leanctx_only_ms = best - interpreter_overhead_ms
    assert leanctx_only_ms <= _BUDGET_MS, (
        f"import leanctx took {leanctx_only_ms:.1f} ms after subtracting "
        f"interpreter overhead {interpreter_overhead_ms:.1f} ms "
        f"(budget {_BUDGET_MS} ms). Raw timings: {timings}"
    )


def _measure_python_overhead(repo_root: Path) -> float:
    """Estimate the cost of forking Python with no imports."""
    samples: list[float] = []
    for _ in range(_RUNS):
        start = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "pass"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
        samples.append((time.perf_counter() - start) * 1000.0)
    return min(samples)
