"""CLI: run the full agent pipeline over a batch of images and print a summary.

    python -m src.agents.run_batch --dir data/raw/casting_data/casting_data/test --limit 10
    python -m src.agents.run_batch --images path/a.jpeg path/b.jpeg

--dir scans recursively, so it works whether pointed directly at a class
folder (.../test/def_front) or at a parent directory containing several
(.../test, holding both def_front/ and ok_front/). When scanning a parent
directory, the result is interleaved round-robin across immediate
subfolders (not just alphabetically sorted) specifically so that --limit
draws from every subfolder rather than exhausting one alphabetically-first
folder (e.g. def_front) before ever reaching the next (ok_front).

Runs each image through the compiled graph sequentially (this codebase's
model/knowledge-base caches -- src.inference.predict, root_cause_agent's
ChromaDB collection -- are module-level singletons built for one process
at a time, not concurrency; sequential is also what makes the Trend
Agent's per-batch simulation meaningful -- see src.agents.trend_agent).
Per-agent failures are already handled inside the graph itself (see
src.agents.orchestrator) -- this runner's own job is just to drive many
images through it and report on the results, so it does not add its own
try/except around graph.invoke().
"""

import argparse
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.graph import get_graph
from src.agents.state import InspectionState
from src.agents.trend_agent import BATCH_IDS, compute_trend

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".bmp"}


def _interleave_by_parent(paths: List[Path]) -> List[Path]:
    """Reorder `paths` round-robin across each path's immediate parent directory.

    A plain recursive scan sorted by full path groups every file from one
    subfolder (e.g. def_front, alphabetically first) before any file from
    the next (ok_front) -- fine for processing an entire directory, but a
    `--limit` cutoff applied to that order would silently only ever see
    the first subfolder. Interleaving means the first N paths, for any N,
    already include a mix from every subfolder present.
    """
    groups: Dict[Path, List[Path]] = {}
    for p in paths:
        groups.setdefault(p.parent, []).append(p)
    for group in groups.values():
        group.sort()

    ordered = []
    for round_paths in zip_longest(*(groups[parent] for parent in sorted(groups))):
        ordered.extend(p for p in round_paths if p is not None)
    return ordered


def collect_image_paths(directory: Optional[str], images: Optional[List[str]]) -> List[Path]:
    """Resolve the --dir/--images CLI args into a concrete list of image paths.

    --dir scans recursively (rglob), not just the top level -- this
    dataset's images live under class subfolders (e.g. test/ok_front/,
    test/def_front/), not directly in the directory a caller is likely to
    point at (e.g. test/) -- and the result is interleaved across those
    subfolders (see _interleave_by_parent) so a --limit cutoff still
    samples a mix rather than one class only. --images is returned in the
    order given, unmodified -- it's an explicit list, not a scan.
    """
    if images:
        return [Path(p) for p in images]
    if directory:
        dir_path = Path(directory)
        found = [p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        return _interleave_by_parent(found)
    raise ValueError("Provide either --dir or --images.")


def run_batch(image_paths: List[Path]) -> List[InspectionState]:
    """Run every image through the graph, sequentially, returning each final state."""
    graph = get_graph()
    return [graph.invoke({"image_path": str(image_path)}) for image_path in image_paths]


def _print_trend_snapshot() -> None:
    """Print the current recent-history trend for every (SIMULATED) batch,
    computed fresh from outputs/logs/inspection_history.jsonl -- i.e. it
    reflects the state of that log after this whole run, not just this
    run's own images (see src.agents.trend_agent module docstring)."""
    print("Final trend snapshot per SIMULATED batch (see src.agents.trend_agent):")
    for batch_id in BATCH_IDS:
        trend = compute_trend(batch_id)
        print(
            f"  {batch_id}: defect_rate={trend.defect_rate:.2f}  scrap_rate={trend.scrap_rate:.2f}  "
            f"drift_flag={trend.drift_flag}  sample_size={trend.sample_size}"
        )


def summarize(results: List[InspectionState]) -> None:
    total = len(results)
    disposition_counts: Counter = Counter()
    human_review_count = 0
    error_images: List[Dict] = []

    for state in results:
        disposition = state.get("disposition")
        if disposition is not None:
            disposition_counts[disposition.decision] += 1
        if state.get("human_review_required"):
            human_review_count += 1
        errors = state.get("errors") or []
        if errors:
            error_images.append({"image_path": state.get("image_path"), "errors": errors})

    print("=" * 72)
    print("BATCH SUMMARY")
    print("=" * 72)
    print(f"Total images processed:     {total}")
    print(f"  accept:                   {disposition_counts.get('accept', 0)}")
    print(f"  rework:                   {disposition_counts.get('rework', 0)}")
    print(f"  scrap:                    {disposition_counts.get('scrap', 0)}")
    print(f"  escalate:                 {disposition_counts.get('escalate', 0)}")
    print(f"  (human_review_required:   {human_review_count})")
    print(f"Images with pipeline errors: {len(error_images)}")
    if error_images:
        print("-" * 72)
        print("Error detail:")
        for entry in error_images:
            for e in entry["errors"]:
                print(f"  {entry['image_path']}: {e['agent']} -> {e['error']}")
    print("-" * 72)
    _print_trend_snapshot()
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Run the agent pipeline over a batch of images.")
    parser.add_argument("--dir", default=None, help="Directory of images to process.")
    parser.add_argument("--images", nargs="*", default=None, help="Explicit list of image paths.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images processed.")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.dir, args.images)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    print(f"Running pipeline on {len(image_paths)} image(s)...")
    results = run_batch(image_paths)
    summarize(results)


if __name__ == "__main__":
    main()
