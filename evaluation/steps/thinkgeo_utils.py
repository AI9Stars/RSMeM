#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the RSMeM evaluation suite.
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_THINKGEO_QUESTION_JSON = "benchmark/thinkgeo_question_eval_ready.json"


def load_json_data(file_path: str) -> Any:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def normalize_question_key(raw_id: Any) -> str:
    text = str(raw_id)
    if text.startswith("question"):
        return text
    if text.isdigit():
        return f"question{text}"
    return text


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).lower().replace(",", "").replace(" ", "").strip()


def infer_question_ids_from_attempt(base_dir: str, round_id: int = 1) -> List[str]:
    attempt_path = Path(base_dir) / f"results_attempt_{round_id}.jsonl"
    if not attempt_path.exists():
        return []

    question_ids: List[str] = []
    with attempt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            question_id = item.get("question_id")
            if question_id is not None:
                question_ids.append(str(question_id))
    return question_ids


def extract_gt_tool_calls(dialogs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for dialog in dialogs:
        if dialog.get("role") != "assistant":
            continue
        for tool_call in dialog.get("tool_calls", []):
            function = tool_call.get("function", {})
            tool_calls.append(
                {
                    "name": function.get("name", ""),
                    "input": function.get("arguments", {}),
                }
            )
    return tool_calls


def select_autonomous_planning_eval(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in evaluations:
        if item.get("type") == "Autonomous Planning":
            return item
    return evaluations[0] if evaluations else {}


def build_thinkgeo_ground_truth(
    question_json_path: str = DEFAULT_THINKGEO_QUESTION_JSON,
    question_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    data = load_json_data(question_json_path)
    if question_ids is None:
        selected_ids = sorted(data.keys(), key=int)
    else:
        selected_ids = [str(qid) for qid in question_ids]

    results: List[Dict[str, Any]] = []
    for qid in selected_ids:
        item = data[str(qid)]
        evaluation_item = select_autonomous_planning_eval(item.get("evaluation", []))
        gt_answer = evaluation_item.get("gt_answer", {}) or {}
        whitelist = [str(x) for x in (gt_answer.get("whitelist") or [])]
        blacklist = [str(x) for x in (gt_answer.get("blacklist") or [])]
        question_text = ""
        for dialog in item.get("dialogs", []):
            if dialog.get("role") == "user":
                question_text = dialog.get("content", "")
                break

        results.append(
            {
                "question_index": normalize_question_key(qid),
                "question_id": str(qid),
                "question": evaluation_item.get("question", question_text),
                "gt_answer": {
                    "whitelist": whitelist,
                    "blacklist": blacklist,
                },
                "final_answer": " | ".join(whitelist),
                "tool_calls": extract_gt_tool_calls(item.get("dialogs", [])),
            }
        )
    return results


def get_predicted_answer_text(item: Dict[str, Any]) -> str:
    return str(
        item.get("final_answer")
        or item.get("polished_answer")
        or item.get("answer")
        or item.get("agent_response")
        or ""
    )


def match_open_ended_answer(predicted_text: Any, gt_answer: Dict[str, Any]) -> bool:
    normalized_prediction = normalize_text(predicted_text)
    whitelist = [normalize_text(x) for x in gt_answer.get("whitelist", []) if normalize_text(x)]
    blacklist = [normalize_text(x) for x in gt_answer.get("blacklist", []) if normalize_text(x)]

    if not whitelist:
        return False

    if any(token in normalized_prediction for token in blacklist):
        return False

    return all(token in normalized_prediction for token in whitelist)


def count_tool_calls(data: Dict[str, Any]) -> int:
    return len(data.get("tool_calls", []))
