#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from thinkgeo_utils import normalize_question_key


def _question_id_from_message(message: Dict[str, Any]) -> Optional[str]:
    name = message.get("name")
    if isinstance(name, str) and name.isdigit():
        return name

    metadata = message.get("metadata") or {}
    question_id = metadata.get("question_id")
    if question_id is not None:
        return str(question_id)

    return None


def _extract_tool_output(content: Any) -> Any:
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            output_items = first.get("output")
            if isinstance(output_items, list) and output_items:
                maybe_text = output_items[0]
                if isinstance(maybe_text, dict) and "text" in maybe_text:
                    return maybe_text["text"]
    return None


def _parse_chat_file(chat_path: Path) -> Dict[str, List[List[Dict[str, Any]]]]:
    conversations: Dict[str, List[List[Dict[str, Any]]]] = {}
    pending_calls: Dict[str, List[Dict[str, Any]]] = {}
    current_occurrence_index: Dict[str, int] = {}

    with chat_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                message = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            question_id = _question_id_from_message(message)
            if question_id is None:
                continue

            question_key = normalize_question_key(question_id)
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                conversations.setdefault(question_key, []).append([])
                current_occurrence_index[question_key] = len(conversations[question_key]) - 1
                pending_calls[question_key] = []
                continue

            if question_key not in conversations or question_key not in current_occurrence_index:
                conversations.setdefault(question_key, []).append([])
                current_occurrence_index[question_key] = len(conversations[question_key]) - 1
                pending_calls[question_key] = []

            if role == "assistant" and isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_use":
                        continue
                    tool_call = {
                        "name": item.get("name", ""),
                        "input": item.get("input", {}),
                        "output": None,
                    }
                    occurrence_idx = current_occurrence_index[question_key]
                    conversations[question_key][occurrence_idx].append(tool_call)
                    pending_calls.setdefault(question_key, []).append(tool_call)

            elif role == "system" and isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_result":
                        continue
                    tool_name = item.get("name", "")
                    output = _extract_tool_output(item.get("output"))
                    for pending in reversed(pending_calls.get(question_key, [])):
                        if pending.get("name") == tool_name and pending.get("output") is None:
                            pending["output"] = output
                            break

    return conversations


def build_tool_call_records(chat_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    conversations = _parse_chat_file(chat_path)
    records_by_question: Dict[str, List[Dict[str, Any]]] = {}

    for question_key, occurrences in conversations.items():
        qid = question_key.replace("question", "")
        occurrences = conversations.get(question_key, [])
        records_by_question.setdefault(question_key, [])
        for tool_calls in occurrences:
            completed_tool_calls = [tc for tc in tool_calls if tc.get("name")]
            records_by_question[question_key].append(
                {
                    "question_index": question_key,
                    "question_id": str(qid),
                    "tool_calls": completed_tool_calls,
                }
            )
    return records_by_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ThinkGeo tool calls from chat logs.")
    parser.add_argument("--base-dir", required=True, help="Run directory containing chat and attempt files")
    parser.add_argument(
        "--chat-file-name",
        default="deepseek-chat_AP_langchain.chat",
        help="Chat log file name under base-dir",
    )
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="Comma-separated round numbers to export",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    chat_path = base_dir / args.chat_file_name
    if not chat_path.exists():
        raise FileNotFoundError(f"Chat file not found: {chat_path}")

    requested_rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]
    records_by_question = build_tool_call_records(chat_path)
    round_occurrence_counter: Dict[str, int] = {}

    for round_id in requested_rounds:
        attempt_file = base_dir / f"results_attempt_{round_id}.jsonl"
        if not attempt_file.exists():
            continue

        round_records: List[Dict[str, Any]] = []
        with attempt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                qid = item.get("question_id")
                if qid is None:
                    continue
                question_key = normalize_question_key(qid)
                occurrence_idx = round_occurrence_counter.get(question_key, 0)
                candidates = records_by_question.get(question_key, [])
                if occurrence_idx < len(candidates):
                    round_records.append(candidates[occurrence_idx])
                else:
                    round_records.append(
                        {
                            "question_index": question_key,
                            "question_id": str(qid),
                            "tool_calls": [],
                        }
                    )
                round_occurrence_counter[question_key] = occurrence_idx + 1

        output_file = base_dir / f"extracted_tool_calls_round{round_id}_thinkgeo.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(round_records, f, ensure_ascii=False, indent=2)
        print(f"Round {round_id}: saved {len(round_records)} questions -> {output_file}")


if __name__ == "__main__":
    main()
