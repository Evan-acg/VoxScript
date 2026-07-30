"""
Performance benchmark for proof subcommand.
Compares current vs new implementation on multiple scenarios.

Usage:
    uv run python scripts/benchmark_proof.py [--iterations 3]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Scenario:
    name: str
    input_file: str
    subtitle_file: str
    flags: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    scenario: str
    total_elapsed: float
    phases: dict[str, float]
    cache_hits: list[str]
    avg_confidence: float | None = None
    error: str | None = None


SCENARIOS: list[Scenario] = [
    Scenario(
        name="same-lang-short",
        input_file="tests/fixtures/audio_short_en.wav",
        subtitle_file="tests/fixtures/subtitle_short_en.srt",
    ),
    Scenario(
        name="translation-short",
        input_file="tests/fixtures/audio_short_en.wav",
        subtitle_file="tests/fixtures/subtitle_short_zh.srt",
        flags=["--llm-check"],
    ),
]

REPORT_TEMPLATE = """
Performance Comparison (median of {runs} runs)
{'=' * 60}
Scenario               Metric               Value        Δ vs baseline
{'─' * 60}
{rows}
"""


def run_scenario(scenario: Scenario, iteration: int, baseline: bool = False) -> BenchmarkResult:
    """Run a single scenario and collect metrics."""
    cmd = [
        sys.executable or "uv", "run", "starter.py", "proof",
        "-i", scenario.input_file,
        "-s", scenario.subtitle_file,
        "-m", "tiny",
        "-f",
        "--output-dir", "./output/benchmark",
        *scenario.flags,
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            scenario=scenario.name,
            total_elapsed=600.0,
            phases={},
            cache_hits=[],
            error="timeout",
        )
    except Exception as e:
        return BenchmarkResult(
            scenario=scenario.name,
            total_elapsed=time.time() - start,
            phases={},
            cache_hits=[],
            error=str(e),
        )

    # Parse report JSON
    report_path = Path("./output/benchmark") / f"{Path(scenario.input_file).stem}.report.json"
    avg_conf = None
    phases = {}
    cache_hits = []

    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            phases = data.get("execution_info", {}).get("phases", {})
            cache_hits = data.get("execution_info", {}).get("cache_hits", [])
        except Exception:
            pass

    return BenchmarkResult(
        scenario=scenario.name,
        total_elapsed=elapsed,
        phases=phases,
        cache_hits=cache_hits,
        avg_confidence=avg_conf,
        error=result.stderr[:500] if result.returncode != 0 else None,
    )


def median(values: list[float]) -> float:
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_v[n // 2]
    return (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2.0


def print_result_table(results: dict[str, list[BenchmarkResult]], iterations: int) -> None:
    print(f"\nBenchmark Results (median of {iterations} runs)")
    print("=" * 70)

    header = f"{'Scenario':<25} {'Total (s)':<12} {'Align (s)':<12} {'ASR (s)':<12} {'LLM (s)':<12}"
    print(header)
    print("-" * 70)

    for scenario_name, runs in results.items():
        totals = [r.total_elapsed for r in runs]
        aligns = [r.phases.get("align", 0) for r in runs]
        asrs = [r.phases.get("whisperx", 0) or r.phases.get("asr", 0) for r in runs]
        llms = [r.phases.get("llm", 0) for r in runs]

        t_med = median(totals)
        a_med = median(aligns)
        asr_med = median(asrs)
        l_med = median(llms)

        errors = [r.error for r in runs if r.error]
        error_mark = " ERROR" if errors else ""

        print(
            f"{scenario_name:<25} {t_med:<12.1f} {a_med:<12.1f} {asr_med:<12.1f} {l_med:<12.1f}{error_mark}"
        )

    print("=" * 70)
    print()

    # Detailed summary for each scenario
    for scenario_name, runs in results.items():
        best = min(runs, key=lambda r: r.total_elapsed)
        print(f"\n--- {scenario_name} (best run) ---")
        print(f"  Total: {best.total_elapsed:.1f}s")
        for phase, dur in sorted(best.phases.items()):
            print(f"  {phase}: {dur:.2f}s")
        if best.cache_hits:
            print(f"  Cache hits: {', '.join(best.cache_hits)}")
        if best.error:
            print(f"  Error: {best.error}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark proof subcommand")
    parser.add_argument("--iterations", type=int, default=3, help="Number of runs per scenario")
    parser.add_argument("--scenario", type=str, default=None, help="Run only specific scenario")
    args = parser.parse_args()

    iterations = args.iterations

    # Ensure output directory exists
    Path("./output/benchmark").mkdir(parents=True, exist_ok=True)

    results: dict[str, list[BenchmarkResult]] = {}

    for scenario in SCENARIOS:
        if args.scenario and args.scenario not in scenario.name:
            continue

        print(f"\nBenchmarking: {scenario.name}")
        scenario_results: list[BenchmarkResult] = []

        for i in range(iterations):
            print(f"  Run {i + 1}/{iterations}...", end=" ", flush=True)
            result = run_scenario(scenario, i)
            scenario_results.append(result)
            status = "OK" if not result.error else f"FAIL: {result.error[:60]}"
            print(f"{status} ({result.total_elapsed:.1f}s)")

        results[scenario.name] = scenario_results

    print_result_table(results, iterations)


if __name__ == "__main__":
    main()
