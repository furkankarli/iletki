"""Summarize the software timestamps produced by the windowed latency replay."""

from collections import Counter, defaultdict
import csv
from math import ceil
from pathlib import Path
import sys


def read(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def quantiles(values):
    ranked = sorted(values)
    return [round(ranked[ceil(len(ranked) * q) - 1], 3) for q in (0.50, 0.95, 0.99)]


def summary(directory, frames, kind, events):
    metrics = defaultdict(list)
    status = Counter()
    seen_frames = set()
    for event in events:
        status[event["status"]] += 1
        frame = frames.get(event["frame"])
        start = int(event["input_ns"])
        phases = {
            "schedule-to-input": (event["scheduled_ns"], start),
            "input-to-pick-encode": (start, event.get("pick_encode_ns", "")),
            "pick-encode-to-submit": (event.get("pick_encode_ns", ""), event["pick_submit_ns"]),
            "pick-submit-to-ready": (event["pick_submit_ns"], event["pick_ready_ns"]),
            "input-to-pick-ready": (start, event["pick_ready_ns"]),
        }
        for label, (a, b) in phases.items():
            if a != "" and b != "":
                metrics[label].append((int(b) - int(a)) / 1_000_000)
        if frame is None:
            continue
        updates = {
            "input-to-frame-start": (start, frame["start_ns"]),
            "input-to-submit": (start, frame["submit_ns"]),
            "input-to-present-return": (start, frame["present_return_ns"]),
            "pick-ready-to-present-return": (event["pick_ready_ns"], frame["present_return_ns"]),
            "pick-ready-to-highlight": (event["pick_ready_ns"], event.get("highlight_applied_ns", "")),
            "redraw-request-to-redraw-event": (event["redraw_ns"], frame["start_ns"]),
        }
        if event.get("first_hit"):
            updates["input-to-contour-frame-return"] = (start, frame["present_return_ns"])
        if event["frame"] not in seen_frames:
            seen_frames.add(event["frame"])
            updates.update({
                "surface-acquire": (frame["start_ns"], frame["acquire_ns"]),
                "encode-after-acquire": (frame["acquire_ns"], frame["encode_ns"]),
                "viewport-encode": (frame["acquire_ns"], frame.get("viewport_encoded_ns", "")),
                "submit-to-observed-gpu": (frame["submit_ns"], frame["gpu_observed_ns"]),
                "present-api": (frame["submit_ns"], frame["present_return_ns"]),
            })
        for label, (a, b) in updates.items():
            if a != "" and b != "":
                metrics[label].append((int(b) - int(a)) / 1_000_000)
    presents = sorted(int(frames[frame]["present_return_ns"]) for frame in seen_frames)
    periods = [(b - a) / 1_000_000 for a, b in zip(presents, presents[1:])]
    metrics["present-return-interval"].extend(periods)
    print(f"{directory}  {kind}  events={len(events)}  status={dict(status)}")
    if len(events) > 1:
        duration = (int(events[-1]["scheduled_ns"]) - int(events[0]["scheduled_ns"])) / 1e9
        duration += duration / (len(events) - 1)
        if duration > 0:
            print(f"  applied previews or inputs per scheduled second: {len(seen_frames) / duration:.2f}")
            submitted = sum(bool(row["pick_submit_ns"]) for row in events)
            if submitted:
                print(f"  picks submitted per scheduled second: {submitted / duration:.2f}")
    counts = [int(row["highlighted"]) for row in events if row.get("highlighted") and row["frame"]]
    if counts:
        print(f"  highlighted objects, min / p50 / max: {min(counts)} / {quantiles(counts)[0]} / {max(counts)}")
    snapshot_flags = [row["preview_view_changed"] for row in events if row.get("preview_view_changed")]
    if snapshot_flags:
        print(f"  applied previews from an earlier view: {snapshot_flags.count('true')}/{len(snapshot_flags)}")
    if periods:
        print(f"  application intervals over 17.7 ms: {sum(x > 17.7 for x in periods)}/{len(periods)}")
    for label, values in metrics.items():
        if values:
            print(f"  {label}: {quantiles(values)} ms p50/p95/p99 ({len(values)} samples)")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for path in sorted(root.rglob("events.csv")):
        frames = {row["frame"]: row for row in read(path.with_name("frames.csv"))}
        kinds = defaultdict(list)
        for row in read(path):
            if row["warmup"] == "false":
                kinds[row["kind"]].append(row)
        for kind, events in kinds.items():
            summary(path.parent, frames, kind, events)
