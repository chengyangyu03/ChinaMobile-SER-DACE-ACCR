#!/usr/bin/env python3
"""Export validation selector output in the exact deployment JSON format."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dace", "accr"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("test_evaluated") is not False:
        raise ValueError("Only validation-only selection output may be exported")
    deployment = selection["deployment_candidate_fit_on_all_validation"]
    if args.method == "accr":
        if deployment.get("primary_calibrator") is None:
            raise ValueError("ACCR requires primary and joint calibrators")
        selection["protocol"] = (
            "outer-4/inner-3 complete-dialogue adaptive confidence-calibrated "
            "routing from step700 primary to optional step1400 secondary"
        )
    elif "coef" not in deployment or len(deployment["coef"][0]) != 6:
        raise ValueError("DACE requires a six-feature joint calibrator")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
