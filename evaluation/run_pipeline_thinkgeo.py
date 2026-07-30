#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_QUESTION_JSON = "benchmark/thinkgeo_question_eval_ready.json"


def _run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _infer_question_ids(base_dir: Path, round_id: int) -> str:
    attempt_path = base_dir / f"results_attempt_{round_id}.jsonl"
    if not attempt_path.exists():
        raise FileNotFoundError(f"Missing attempt file: {attempt_path}")
    question_ids = []
    with attempt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            import json

            item = json.loads(line)
            qid = item.get("question_id")
            if qid is not None:
                question_ids.append(str(qid))
    return ",".join(question_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description="ThinkGeo-specific evaluation pipeline.")
    parser.add_argument("--base-dir", required=True, help="Run directory containing results_attempt_*.jsonl")
    parser.add_argument("--rounds", default="1,2,3", help="Comma-separated rounds")
    parser.add_argument("--result-dir-name", required=True, help="Folder name under results/")
    parser.add_argument("--question-json", default=DEFAULT_QUESTION_JSON, help="ThinkGeo aligned question file")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rounds = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    question_ids = _infer_question_ids(base_dir, rounds[0])

    py = sys.executable
    here = Path(__file__).resolve().parent
    steps = here / "steps"

    _run(
        [
            py,
            str(steps / "prepare_complete_results.py"),
            "--base-dir",
            str(base_dir),
            "--rounds",
            ",".join(str(r) for r in rounds),
        ]
    )
    _run(
        [
            py,
            str(steps / "prepare_extract_tool_calls_thinkgeo.py"),
            "--base-dir",
            str(base_dir),
            "--rounds",
            ",".join(str(r) for r in rounds),
        ]
    )
    _run(
        [
            py,
            str(steps / "prepare_merge_tool_calls_thinkgeo.py"),
            "--base-dir",
            str(base_dir),
            "--rounds",
            ",".join(str(r) for r in rounds),
        ]
    )

    _run(
        [
            py,
            str(steps / "metric_step_by_step_thinkgeo.py"),
            "--base-dir",
            str(base_dir),
            "--question-json",
            args.question_json,
            "--question-ids",
            question_ids,
            "--rounds",
            ",".join(str(r) for r in rounds),
            "--result-dir-name",
            args.result_dir_name,
        ]
    )
    _run(
        [
            py,
            str(steps / "metric_end_to_end_thinkgeo.py"),
            "--base-dir",
            str(base_dir),
            "--question-json",
            args.question_json,
            "--question-ids",
            question_ids,
            "--rounds",
            ",".join(str(r) for r in rounds),
            "--result-dir-name",
            args.result_dir_name,
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
