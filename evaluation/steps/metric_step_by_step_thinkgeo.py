#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from thinkgeo_utils import (
    build_thinkgeo_ground_truth,
    load_json_data,
    normalize_question_key,
)


GT_TOOL_MAP = {
    "SegmentObjectPixels": "SEGMENT",
    "TextToBbox": "DETECT",
    "ObjectDetection": "DETECT",
    "CountGivenObject": "COUNT",
    "Calculator": "MATH",
    "Solver": "MATH",
}

RUNTIME_TOOL_MAP = {
    "get_filelist": None,
    "calculate_area": "SEGMENT",
    "threshold_segmentation": "SEGMENT",
    "InstructSAM": "SEGMENT",
    "RemoteSAM": "SEGMENT",
    "SAM2": "SEGMENT",
    "ChangeOS": "SEGMENT",
    "count_above_threshold": "COUNT",
    "count_skeleton_contours": "COUNT",
    "SM3Det": "DETECT",
    "Strip_R_CNN": "DETECT",
    "RemoteCLIP": "DETECT",
    "get_list_object_via_indexes": "DETECT",
    "calculate_bbox_area": "GEOMETRY",
    "bboxes2centroids": "GEOMETRY",
    "centroid_distance_extremes": "GEOMETRY",
    "mean": "MATH",
    "multiply": "MATH",
    "division": "MATH",
    "difference": "MATH",
    "ceil_number": "MATH",
    "percentage_change": "MATH",
    "calc_batch_image_min": "STATS",
    "calc_batch_image_max": "STATS",
    "calc_batch_image_mean": "STATS",
    "calc_batch_image_median": "STATS",
    "calc_batch_image_sum": "STATS",
    "calc_batch_image_hotspot_percentage": "STATS",
    "calc_batch_image_hotspot_tif": "STATS",
}


def canonical_tool_name(name: str, source: str) -> Optional[str]:
    if source == "gt":
        return GT_TOOL_MAP.get(name, name)
    return RUNTIME_TOOL_MAP.get(name, name)


def _flatten_input_tokens(value: Any) -> List[str]:
    tokens: List[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            tokens.extend(_flatten_input_tokens(nested))
    elif isinstance(value, list):
        for nested in value:
            tokens.extend(_flatten_input_tokens(nested))
    elif isinstance(value, (int, float)):
        tokens.append(f"{float(value):.3f}")
    elif isinstance(value, str):
        parts = re.split(r"[^a-zA-Z0-9._/-]+", value.lower())
        for part in parts:
            if not part:
                continue
            if "/" in part:
                part = part.split("/")[-1]
            tokens.append(part)
    return tokens


def canonicalize_tool_calls(tool_calls: List[Dict], source: str) -> List[Dict]:
    canonicalized: List[Dict] = []
    for call in tool_calls:
        name = call.get("name", "")
        mapped_name = canonical_tool_name(name, source)
        if mapped_name is None:
            continue
        item = {
            "name": mapped_name,
            "raw_name": name,
            "input": call.get("input", {}),
            "tokens": set(_flatten_input_tokens(call.get("input", {}))),
        }
        if canonicalized and canonicalized[-1]["name"] == item["name"]:
            canonicalized[-1]["tokens"].update(item["tokens"])
            continue
        canonicalized.append(item)
    return canonicalized


def extract_tool_names_from_calls(tool_calls: List[Dict], source: str) -> List[str]:
    return [call["name"] for call in canonicalize_tool_calls(tool_calls, source)]


def contains_all_tool_calls_any_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    expected_tools = extract_tool_names_from_calls(ground_truth_data.get("tool_calls", []), "gt")
    actual_tools = extract_tool_names_from_calls(predicted_data.get("tool_calls", []), "runtime")
    if not expected_tools:
        return {"score": 1.0, "key": "contains_all_tool_calls_any_order"}
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    matched = expected_set.intersection(actual_set)
    return {
        "score": len(matched) / len(expected_set),
        "key": "contains_all_tool_calls_any_order",
    }


def contains_all_tool_calls_in_order(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    expected_tools = extract_tool_names_from_calls(ground_truth_data.get("tool_calls", []), "gt")
    actual_tools = extract_tool_names_from_calls(predicted_data.get("tool_calls", []), "runtime")
    if not expected_tools:
        return {"score": 1.0, "key": "contains_all_tool_calls_in_order"}
    matched_count = 0
    actual_iter = iter(actual_tools)
    for expected_tool in expected_tools:
        found = False
        for actual_tool in actual_iter:
            if actual_tool == expected_tool:
                matched_count += 1
                found = True
                break
        if not found:
            break
    return {
        "score": matched_count / len(expected_tools),
        "key": "contains_all_tool_calls_in_order",
    }


def contains_all_tool_calls_in_order_exact_match(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    expected_tools = extract_tool_names_from_calls(ground_truth_data.get("tool_calls", []), "gt")
    actual_tools = extract_tool_names_from_calls(predicted_data.get("tool_calls", []), "runtime")
    if not expected_tools:
        return {
            "score": 1.0 if not actual_tools else 0.0,
            "key": "contains_all_tool_calls_in_order_exact_match",
        }
    min_length = min(len(expected_tools), len(actual_tools))
    exact_matches = 0
    for i in range(min_length):
        if expected_tools[i] == actual_tools[i]:
            exact_matches += 1
        else:
            break
    total_positions = max(len(expected_tools), len(actual_tools))
    return {
        "score": exact_matches / total_positions if total_positions else 1.0,
        "key": "contains_all_tool_calls_in_order_exact_match",
    }


def trajectory_step_wise_score(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    expected_tools = extract_tool_names_from_calls(ground_truth_data.get("tool_calls", []), "gt")
    actual_tools = extract_tool_names_from_calls(predicted_data.get("tool_calls", []), "runtime")
    if not expected_tools:
        return {"score": 0.0, "key": "trajectory_step_wise"}
    correct_steps = 0
    for idx in range(min(len(expected_tools), len(actual_tools))):
        if expected_tools[idx] == actual_tools[idx]:
            correct_steps += 1
        else:
            break
    return {
        "score": correct_steps / len(expected_tools),
        "key": "trajectory_step_wise",
    }


def check_parameter_accuracy(predicted_data: Dict, ground_truth_data: Dict) -> dict:
    predicted_calls = canonicalize_tool_calls(predicted_data.get("tool_calls", []), "runtime")
    expected_calls = canonicalize_tool_calls(ground_truth_data.get("tool_calls", []), "gt")
    if not expected_calls:
        return {"score": 1.0, "key": "parameter_accuracy"}
    matched_steps = 0
    for idx in range(min(len(predicted_calls), len(expected_calls))):
        pred_call = predicted_calls[idx]
        exp_call = expected_calls[idx]
        if pred_call.get("name") != exp_call.get("name"):
            break

        if pred_call["name"] in {"MATH", "GEOMETRY", "COUNT", "STATS"}:
            matched_steps += 1
            continue

        overlap = pred_call.get("tokens", set()).intersection(exp_call.get("tokens", set()))
        if overlap:
            matched_steps += 1
        else:
            break
    return {
        "score": matched_steps / len(expected_calls),
        "key": "parameter_accuracy",
    }


def evaluate_single_question(predicted_data: Dict, ground_truth_data: Dict) -> Dict:
    return {
        "contains_all_tool_calls_any_order": contains_all_tool_calls_any_order(predicted_data, ground_truth_data),
        "contains_all_tool_calls_in_order": contains_all_tool_calls_in_order(predicted_data, ground_truth_data),
        "contains_all_tool_calls_in_order_exact_match": contains_all_tool_calls_in_order_exact_match(predicted_data, ground_truth_data),
        "trajectory_step_wise_score": trajectory_step_wise_score(predicted_data, ground_truth_data),
        "parameter_accuracy": check_parameter_accuracy(predicted_data, ground_truth_data),
    }


def run_step_by_step_evaluation(predicted_file: str, question_json: str, question_ids: List[str]) -> Dict:
    predicted_data = load_json_data(predicted_file)
    ground_truth_data = build_thinkgeo_ground_truth(question_json, question_ids=question_ids)
    gt_dict = {item["question_index"]: item for item in ground_truth_data}
    pred_dict = {normalize_question_key(item["question_index"]): item for item in predicted_data}

    all_results = {}
    summary_stats = {
        "total_questions": len(gt_dict),
        "evaluated_questions": 0,
        "missing_predictions": [],
        "metrics_summary": {
            "contains_all_tool_calls_any_order": {"total_score": 0, "count": 0},
            "contains_all_tool_calls_in_order": {"total_score": 0, "count": 0},
            "contains_all_tool_calls_in_order_exact_match": {"total_score": 0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0, "count": 0},
            "parameter_accuracy": {"total_score": 0, "count": 0},
        },
    }

    for question_index, gt_item in gt_dict.items():
        if question_index not in pred_dict:
            summary_stats["missing_predictions"].append(question_index)
            continue
        pred_item = pred_dict[question_index]
        summary_stats["evaluated_questions"] += 1
        question_results = evaluate_single_question(pred_item, gt_item)
        all_results[question_index] = question_results
        for metric_name, metric_result in question_results.items():
            summary_stats["metrics_summary"][metric_name]["total_score"] += metric_result["score"]
            summary_stats["metrics_summary"][metric_name]["count"] += 1

    for metric_name, metric_stats in summary_stats["metrics_summary"].items():
        count = metric_stats["count"]
        metric_stats["average_score"] = metric_stats["total_score"] / count if count else 0.0

    return {"individual_results": all_results, "summary": summary_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkGeo step-by-step evaluation.")
    parser.add_argument("--base-dir", required=True, help="Directory containing extracted_tool_calls_round{N}_thinkgeo.json")
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
        predicted_file = Path(args.base_dir) / f"merged_tool_calls_round{round_id}_thinkgeo.json"
        if not predicted_file.exists():
            print(f"Skipping round {round_id}: {predicted_file} not found")
            continue
        results = run_step_by_step_evaluation(str(predicted_file), args.question_json, question_ids)
        output_file = output_dir / f"step_by_step_evaluation_results_round{round_id}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Round {round_id} saved to {output_file}")


if __name__ == "__main__":
    main()
