from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import DATASET_ORDER, OUTPUTS_DIR, get_single_norm_curve_file
from src.measures.focus_measure_library import build_focus_measure_registry


def main() -> None:
    failures: list[str] = []
    registry = build_focus_measure_registry()
    if len(registry) != 32:
        failures.append(f"Expected 32 active measures, found {len(registry)}")

    expected_curves = [
        get_single_norm_curve_file(dataset, measure)
        for dataset in DATASET_ORDER
        for measure in registry
    ]
    missing_curves = [str(path.relative_to(ROOT)) for path in expected_curves if not path.exists()]
    if missing_curves:
        failures.append(f"Missing normalized curves: {len(missing_curves)}")

    forbidden = OUTPUTS_DIR / "01_stacks" / "arrays"
    if forbidden.exists() and any(forbidden.iterdir()):
        failures.append("The excluded outputs/01_stacks/arrays cache is present")

    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"Invalid JSON {path.relative_to(ROOT)}: {exc}")

    for path in ROOT.rglob("*.csv"):
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                list(csv.reader(handle))
        except Exception as exc:
            failures.append(f"Unreadable CSV {path.relative_to(ROOT)}: {exc}")

    required = [
        ROOT / "README.md",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "outputs" / "09_paper" / "manifests" / "asset_index.json",
        ROOT / "outputs" / "10_review_response_computations" / "review_response_manifest.json",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"Missing required file: {path.relative_to(ROOT)}")

    if failures:
        print("Release validation: FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Release validation: PASSED")
    print(f"- datasets: {', '.join(DATASET_ORDER)}")
    print(f"- active measures: {len(registry)}")
    print(f"- normalized curve files: {len(expected_curves)}")
    print("- reconstructed stack arrays included: no")


if __name__ == "__main__":
    main()
