#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, List

from thinkgeo_utils import (
    build_thinkgeo_ground_truth,
    count_tool_calls,
    get_predicted_answer_text,
    load_json_data,
    match_open_ended_answer,
    normalize_question_key,
)


def calculate_accuracy(ground_truth_data: List[Dict], predicted_data: Dict) -> Dict:
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    if isinstance(predicted_data, list):
        pred_dict = {
            normalize_question_key(item.get("question_id") or item.get("question_index")): item
            for item in predicted_data
            if item.get("question_id") is not None or item.get("question_index") is not None
        }
    else:
        pred_dict = {normalize_question_key(k): v for k, v in predicted_data.items()}

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "fail_answers": 0,
        "unknown_answers": 0,
        "missing_predictions": [],
        "accuracy": 0.0,
        "detailed_results": [],
    }

    for question_index, gt_item in gt_dict.items():
        if question_index not in pred_dict:
            results["missing_predictions"].append(question_index)
            results["detailed_results"].append(
                {
                    "question_index": question_index,
                    "ground_truth": gt_item.get("gt_answer", {}),
                    "predicted": "MISSING",
                    "correct": False,
                    "status": "missing",
                }
            )
            continue

        pred_item = pred_dict[question_index]
        results["evaluated_questions"] += 1
        answer_text = get_predicted_answer_text(pred_item)
        is_correct = match_open_ended_answer(answer_text, gt_item.get("gt_answer", {}))
        if is_correct:
            results["correct_answers"] += 1
            status = "correct"
        elif not answer_text.strip():
            results["fail_answers"] += 1
            status = "fail"
        else:
            status = "incorrect"

        results["detailed_results"].append(
            {
                "question_index": question_index,
                "ground_truth": gt_item.get("gt_answer", {}),
                "predicted": answer_text,
                "correct": is_correct,
                "status": status,
            }
        )

    if results["evaluated_questions"] > 0:
        results["accuracy"] = results["correct_answers"] / results["evaluated_questions"]
    return results


def calculate_efficiency_with_tool_calls(ground_truth_data: List[Dict], model_tool_calls_data: List[Dict]) -> Dict:
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    model_dict = {normalize_question_key(item["question_index"]): item for item in model_tool_calls_data}

    results = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "average_efficiency": 0.0,
        "detailed_results": [],
    }

    for question_index, gt_item in gt_dict.items():
        gt_tool_count = count_tool_calls(gt_item)
        if question_index not in model_dict:
            results["detailed_results"].append(
                {
                    "question_index": question_index,
                    "gt_tool_count": gt_tool_count,
                    "model_tool_count": 0,
                    "efficiency": 0.0,
                    "status": "missing",
                }
            )
            continue

        model_item = model_dict[question_index]
        model_tool_count = count_tool_calls(model_item)
        results["evaluated_questions"] += 1

        if gt_tool_count == 0:
            efficiency = 1.0 if model_tool_count == 0 else float("inf")
        else:
            efficiency = model_tool_count / gt_tool_count

        results["efficiency_scores"].append(efficiency)
        results["detailed_results"].append(
            {
                "question_index": question_index,
                "gt_tool_count": gt_tool_count,
                "model_tool_count": model_tool_count,
                "efficiency": efficiency,
                "status": "evaluated",
            }
        )

    if results["efficiency_scores"]:
        finite_scores = [x for x in results["efficiency_scores"] if x != float("inf")]
        results["average_efficiency"] = sum(finite_scores) / len(finite_scores) if finite_scores else float("inf")
    return results


def run_end_to_end_evaluation(
    question_json: str,
    question_ids: List[str],
    predicted_answers_file: str,
    model_tool_calls_file: str,
) -> Dict:
    ground_truth_data = build_thinkgeo_ground_truth(question_json, question_ids=question_ids)
    predicted_data = load_json_data(predicted_answers_file)
    model_tool_calls_data = load_json_data(model_tool_calls_file)

    accuracy_results = calculate_accuracy(ground_truth_data, predicted_data)
    efficiency_results = calculate_efficiency_with_tool_calls(ground_truth_data, model_tool_calls_data)

    return {
        "accuracy": accuracy_results,
        "efficiency": efficiency_results,
        "summary": {
            "total_questions": accuracy_results["total_questions"],
            "accuracy_rate": accuracy_results["accuracy"],
            "average_efficiency": efficiency_results.get("average_efficiency", 0.0),
            "fail_rate": accuracy_results["fail_answers"] / accuracy_results["evaluated_questions"]
            if accuracy_results["evaluated_questions"] > 0
            else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkGeo end-to-end evaluation.")
    parser.add_argument("--base-dir", required=True, help="Directory containing result files")
    parser.add_argument("--question-json", required=True, help="ThinkGeo question json")
    parser.add_argument("--question-ids", required=True, help="Comma-separated question ids to evaluate")
    parser.add_argument("--rounds", default="1,2,3", help="Comma-separated round numbers")
    parser.add_argument("--result-dir-name", required=True, help="Folder name under results/")
    args = parser.parse_args()

    question_ids = [x.strip() for x in args.question_ids.split(",") if x.strip()]
    rounds = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    output_dir = Path(__file__).resolve().parents[1] / "results" / args.result_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for round_id in rounds:
        predicted_answers_file = Path(args.base_dir) / f"results_summary_ROUND{round_id}.jsonl"
        model_tool_calls_file = Path(args.base_dir) / f"merged_tool_calls_round{round_id}_thinkgeo.json"
        if not predicted_answers_file.exists() or not model_tool_calls_file.exists():
            print(f"Skipping round {round_id}: missing inputs")
            continue
        results = run_end_to_end_evaluation(
            args.question_json,
            question_ids,
            str(predicted_answers_file),
            str(model_tool_calls_file),
        )
        output_file = output_dir / f"end_to_end_evaluation_results_round{round_id}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Round {round_id} saved to {output_file}")


if __name__ == "__main__":
    main()
