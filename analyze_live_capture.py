#!/usr/bin/env python3
"""Summarize one live-session capture produced by server.py."""

from __future__ import annotations

import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path


def latest_capture(root: Path) -> Path:
    captures = sorted([p for p in root.glob("live_*") if p.is_dir()])
    if not captures:
        raise SystemExit(f"No captures found in {root}")
    return captures[-1]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_capture(Path("logs/live_sessions"))
    if root.is_dir():
        steps_path = root / "steps.ndjson"
    else:
        steps_path = root
        root = steps_path.parent

    rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No steps in {steps_path}")

    by_label = Counter()
    rms_by_label = defaultdict(list)
    peak_by_label = defaultdict(list)
    onsets = []
    unknown_flat = []
    prev_present = set()

    for i, row in enumerate(rows):
        speakers = row.get("speakers", [])
        present = {int(sp["id"]) for sp in speakers}
        by_id = {int(sp["id"]): sp for sp in speakers}

        for sp in speakers:
            label = str(sp["label"])
            by_label[label] += 1
            rms_by_label[label].append(float(sp.get("audio_rms", 0.0)))
            peak_by_label[label].append(float(sp.get("audio_peak", 0.0)))
            if label.startswith("Unknown-") and float(sp.get("audio_rms", 0.0)) < 0.02:
                unknown_flat.append((row["step"], label, sp.get("audio_rms", 0.0)))

        new_ids = sorted(present - prev_present)
        for speaker_id in new_ids:
            cur = by_id[speaker_id]
            nxt = None
            if i + 1 < len(rows):
                for sp in rows[i + 1].get("speakers", []):
                    if int(sp["id"]) == speaker_id:
                        nxt = sp
                        break
            if nxt is not None:
                cur_rms = float(cur.get("audio_rms", 0.0))
                nxt_rms = float(nxt.get("audio_rms", 0.0))
                ratio = cur_rms / (nxt_rms + 1e-12)
                onsets.append((row["step"], cur["label"], cur_rms, nxt_rms, ratio))
        prev_present = present

    print(f"capture: {root}")
    print(f"steps: {len(rows)}")
    print()
    print("speaker emissions:")
    for label, count in by_label.most_common():
        mean_rms = stats.fmean(rms_by_label[label]) if rms_by_label[label] else 0.0
        mean_peak = stats.fmean(peak_by_label[label]) if peak_by_label[label] else 0.0
        print(f"  {label:>10s}: steps={count:3d} mean_rms={mean_rms:.4f} mean_peak={mean_peak:.4f}")

    print()
    print("onset attack ratios (current_step_rms / next_step_rms):")
    for step, label, cur_rms, nxt_rms, ratio in onsets[:20]:
        print(f"  step={step:3d} {label:>10s} cur={cur_rms:.4f} next={nxt_rms:.4f} ratio={ratio:.3f}")

    print()
    print(f"flat unknown outputs (<0.02 rms): {len(unknown_flat)}")
    for step, label, rms in unknown_flat[:20]:
        print(f"  step={step:3d} {label:>10s} rms={rms:.4f}")


if __name__ == "__main__":
    main()
