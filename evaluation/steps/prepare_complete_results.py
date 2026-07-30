# Adapted from the Earth-Agent evaluation framework
# (https://github.com/opendatalab/Earth-Agent), MIT License,
# Copyright (c) 2026 OpenDataLab. See the NOTICE file at the repo root.
import argparse
import json
import os
import shutil
from typing import Dict, List, Optional


def _load_jsonl_as_map(path: str) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = item.get("question_id")
            if qid is None:
                continue
            data[str(qid)] = item
    return data


def _infer_rounds(base_dir: str, max_rounds: int = 20) -> List[int]:
    rounds: List[int] = []
    for r in range(1, max_rounds + 1):
        if os.path.exists(os.path.join(base_dir, f"results_attempt_{r}.jsonl")):
            rounds.append(r)
    return rounds


def complete_results(base_dir: str, rounds: Optional[List[int]] = None) -> List[int]:
    """， results_summary_ROUND{r}.jsonl。

    （ r）：
    - ROUND1:  results_attempt_1.jsonl
    - ROUND r (r>=2):  question_id， attempt_r； attempt_{r-1} ...  attempt_1。
    """

    if rounds is None or len(rounds) == 0:
        rounds = _infer_rounds(base_dir)

    rounds = sorted(set(int(x) for x in rounds if int(x) >= 1))
    if not rounds:
        print("Error: no rounds specified/inferred.")
        return []

    file1 = os.path.join(base_dir, "results_attempt_1.jsonl")
    if not os.path.exists(file1):
        print(f"Error: results_attempt_1.jsonl missing: {file1}")
        return []

    print(f"Loading files from {base_dir}...")

    attempt_data: Dict[int, Dict[str, dict]] = {}
    for r in rounds:
        attempt_path = os.path.join(base_dir, f"results_attempt_{r}.jsonl")
        if not os.path.exists(attempt_path):
            print(f"Warning: File not found, will skip generating ROUND{r}: {attempt_path}")
            continue
        attempt_data[r] = _load_jsonl_as_map(attempt_path)
        if not attempt_data[r]:
            print(f"Warning: File is empty/unreadable, will skip generating ROUND{r}: {attempt_path}")
            attempt_data.pop(r, None)

    if 1 not in attempt_data:
        attempt_data[1] = _load_jsonl_as_map(file1)
    if not attempt_data[1]:
        print("Error: results_attempt_1.jsonl is empty or unreadable. Cannot proceed.")
        return []

    all_ids = sorted(attempt_data[1].keys(), key=lambda x: int(x))
    print(f"Total questions identified: {len(all_ids)}")

    generated_rounds: List[int] = []
    for r in rounds:
        if r == 1:
            output_path = os.path.join(base_dir, "results_summary_ROUND1.jsonl")
            shutil.copyfile(file1, output_path)
            generated_rounds.append(1)
            print(f"  - Attempt 1 (copied): {len(all_ids)} items -> {output_path}")
            continue

        if r not in attempt_data:
            continue

        completed: List[dict] = []
        for qid in all_ids:
            chosen_item = None
            for back in range(r, 0, -1):
                data = attempt_data.get(back)
                if data and qid in data:
                    chosen_item = data[qid]
                    break
            if chosen_item is None:
                chosen_item = attempt_data[1][qid]
            completed.append(chosen_item)

        output_path = os.path.join(base_dir, f"results_summary_ROUND{r}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in completed:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        generated_rounds.append(r)
        print(f"  - Attempt {r} (completed): {len(completed)} items -> {output_path}")

    if not generated_rounds:
        print("Warning: no results_summary_ROUND*.jsonl generated.")
    return generated_rounds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete multi-round results by filling missing IDs and generating ROUND summaries.")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory containing results_attempt_{N}.jsonl",
    )
    parser.add_argument(
        "--rounds",
        default=None,
        help="Comma-separated round numbers to generate, e.g. 1,2,3,4,5. Default: infer from existing results_attempt_*.jsonl",
    )
    args = parser.parse_args()

    rounds = None
    if args.rounds:
        rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]

    complete_results(args.base_dir, rounds=rounds)
