#!/usr/bin/env python3
"""Summarize per-step inference traces from logs/infer_step_perf.ndjson."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "step_perf":
                rows.append(row)
    return rows


def _mean(vals: list[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def _pctl(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return float(ordered[idx])


def _fmt_ms(val: float) -> str:
    return f"{val:.1f}ms"


def main() -> int:
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/infer_step_perf.ndjson")
    path = raw_path if raw_path.is_absolute() else Path.cwd() / raw_path
    if not path.exists():
        print(f"Log file not found: {path}")
        return 1

    rows = _load_rows(path)
    if not rows:
        print(f"No step_perf rows found in {path}")
        return 1

    infer = [float(r["infer_ms"]) for r in rows]
    spike_threshold = max(340.0, _pctl(infer, 0.9))
    spikes = [r for r in rows if float(r["infer_ms"]) >= spike_threshold]

    print(f"file: {path}")
    print(f"steps: {len(rows)}")
    print(
        "infer: "
        f"mean={_fmt_ms(_mean(infer))} "
        f"median={_fmt_ms(statistics.median(infer))} "
        f"p95={_fmt_ms(_pctl(infer, 0.95))} "
        f"max={_fmt_ms(max(infer))}"
    )
    print(f"slow-step threshold: {_fmt_ms(spike_threshold)} ({len(spikes)} slow steps)")

    stage_keys = [
        "input_h2d",
        "pixit",
        "seg_tail_sync",
        "embed_prep",
        "embed_forward",
        "cluster",
        "aggregate",
        "map",
        "extract",
    ]
    print("\nstage means:")
    for key in stage_keys:
        vals = [float(r["timings_ms"].get(key, 0.0)) for r in rows]
        print(f"  {key:>14}: {_fmt_ms(_mean(vals))}")

    if spikes:
        print("\nslow-step deltas vs overall:")
        for key in [
            "embed_prep",
            "embed_forward",
            "map",
            "extract",
            "pixit",
            "seg_tail_sync",
        ]:
            slow_mean = _mean([float(r["timings_ms"].get(key, 0.0)) for r in spikes])
            all_mean = _mean([float(r["timings_ms"].get(key, 0.0)) for r in rows])
            print(f"  {key:>14}: {_fmt_ms(slow_mean)} (all {_fmt_ms(all_mean)})")
        for key in [
            "active_locals",
            "prepared_sources",
            "full_buffer_fallback_hits",
            "anchor_checks",
            "holdover_added",
            "final_pairs",
        ]:
            slow_mean = _mean([float(r["counts"].get(key, 0.0)) for r in spikes])
            all_mean = _mean([float(r["counts"].get(key, 0.0)) for r in rows])
            print(f"  {key:>14}: {slow_mean:.2f} (all {all_mean:.2f})")

    print("\nworst steps:")
    for row in sorted(rows, key=lambda r: float(r["infer_ms"]), reverse=True)[:12]:
        t = row["timings_ms"]
        c = row["counts"]
        print(
            "  "
            f"step={row['step']:>4} "
            f"infer={_fmt_ms(float(row['infer_ms']))} "
            f"pixit={_fmt_ms(float(t.get('pixit', 0.0)))} "
            f"seg_sync={_fmt_ms(float(t.get('seg_tail_sync', 0.0)))} "
            f"embed={_fmt_ms(float(t.get('embed_prep', 0.0)) + float(t.get('embed_forward', 0.0)))} "
            f"map={_fmt_ms(float(t.get('map', 0.0)))} "
            f"extract={_fmt_ms(float(t.get('extract', 0.0)))} "
            f"active={int(c.get('active_locals', 0))} "
            f"prep={int(c.get('prepared_sources', 0))} "
            f"fallback={int(c.get('full_buffer_fallback_hits', 0))} "
            f"checks={int(c.get('anchor_checks', 0))} "
            f"holdover={int(c.get('holdover_added', 0))} "
            f"pairs={int(c.get('final_pairs', 0))}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
