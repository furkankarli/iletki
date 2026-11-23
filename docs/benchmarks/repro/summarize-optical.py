"""Check synchronized input-to-display trials against an iletki window trace."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from math import ceil
from pathlib import Path
import sys


LIMITS_NS = {
    "camera": (33_300_000, 50_000_000),
    "click": (50_000_000, 66_700_000),
    "marquee-release": (50_000_000, 66_700_000),
}
CAMERA_EVENTS = {"drag", "zoom", "orbit", "pan"}


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def frame_marker(frame):
    return (frame ^ (frame >> 1)) & 0xFFFF


def presentation_frames(path):
    frames = defaultdict(list)
    for row in read_rows(path):
        if row["status"] == "present-called":
            frame = int(row["frame"])
            frames[frame_marker(frame)].append(frame)
    return frames


def observations(trace, capture):
    events = {int(row["event"]): row for row in read_rows(trace / "events.csv")}
    frames = presentation_frames(trace / "frames.csv")
    boundaries = sorted({int(row["frame"]) for row in events.values() if row["frame"]})
    seen = set()
    for row in read_rows(capture):
        event_id = int(row["event"])
        if event_id in seen:
            raise ValueError(f"duplicate trial for event {event_id}")
        seen.add(event_id)
        if event_id not in events:
            raise ValueError(f"unknown trace event {event_id}")
        event = events[event_id]
        if event["warmup"] == "true":
            continue
        kind = "camera" if event["kind"] in CAMERA_EVENTS else event["kind"]
        if kind not in LIMITS_NS:
            raise ValueError(f"event {event_id} is not a camera, click or release trial")

        trigger = int(row["trigger_ns"])
        if trigger < 0:
            raise ValueError(f"event {event_id} has a negative trigger time")
        if not row["changed_ns"] and not row["gray16"]:
            yield kind, "no-visible-change", None
            continue
        changed = int(row["changed_ns"])
        if changed < trigger:
            raise ValueError(f"event {event_id} changes before its trigger")
        marker = row["gray16"].strip().lower()
        observed = int(marker, 16 if marker.startswith("0x") else 10)
        if not 0 <= observed <= 0xFFFF:
            raise ValueError(f"event {event_id} has a marker outside the 16-bit range")
        if not event["frame"]:
            yield kind, "no-application-frame", None
            continue

        first = int(event["frame"])
        next_event = bisect_right(boundaries, first)
        following = boundaries[next_event] if next_event < len(boundaries) else None
        candidates = [
            frame for frame in frames.get(observed, ())
            if frame >= first and (following is None or frame < following)
        ]
        if len(candidates) != 1:
            yield kind, "marker-mismatch", None
            continue
        yield kind, "matched", changed - trigger


def quantile(values, percentile):
    return values[ceil(len(values) * percentile) - 1]


def run(trace, capture, minimum):
    groups = defaultdict(Counter)
    latencies = defaultdict(list)
    for kind, status, delay in observations(trace, capture):
        groups[kind][status] += 1
        if delay is not None:
            latencies[kind].append(delay)
    if not groups:
        raise ValueError("capture has no measured trials")

    passed = True
    for kind, counts in sorted(groups.items()):
        values = sorted(latencies[kind])
        p95_limit, p99_limit = LIMITS_NS[kind]
        grades = len(values) >= minimum and len(values) == counts.total()
        result = "no matched trials"
        if values:
            p95, p99 = quantile(values, .95), quantile(values, .99)
            grades &= p95 <= p95_limit and p99 <= p99_limit
            result = f"{p95 / 1e6:.2f}/{p99 / 1e6:.2f} ms p95/p99"
        print(f"{kind}  {result}  {dict(counts)}  {'PASS' if grades else 'FAIL'}")
        passed &= grades
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="directory containing events.csv and frames.csv")
    parser.add_argument("capture", type=Path, help="event,trigger_ns,changed_ns,gray16 CSV")
    parser.add_argument("--min-trials", type=int, default=100, help="minimum per measured action")
    args = parser.parse_args()
    if args.min_trials <= 0:
        parser.error("--min-trials must be positive")
    try:
        return run(args.trace, args.capture, args.min_trials)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"optical capture error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
