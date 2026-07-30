#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def merge_consecutive_same_tools(tool_calls):
    if not tool_calls:
        return []

    merged = []
    i = 0
    while i < len(tool_calls):
        current_tool = tool_calls[i]
        j = i + 1
        while j < len(tool_calls) and tool_calls[j]["name"] == current_tool["name"]:
            j += 1

        if j - i == 1:
            merged.append(current_tool)
        else:
            group = tool_calls[i:j]
            merged_tool = {
                "name": group[0]["name"],
                "input": OrderedDict(),
                "output": [tool.get("output") for tool in group],
            }
            if isinstance(group[0].get("input"), dict):
                for key in group[0]["input"].keys():
                    merged_tool["input"][key] = [
                        tool["input"][key]
                        for tool in group
                        if isinstance(tool.get("input"), dict) and key in tool["input"]
                    ]
            merged.append(merged_tool)
        i = j
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ThinkGeo extracted tool calls.")
    parser.add_argument("--base-dir", required=True, help="Directory containing extracted_tool_calls_round{N}_thinkgeo.json")
    parser.add_argument("--rounds", default="1,2,3", help="Comma-separated rounds")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]

    for round_id in rounds:
        input_file = base_dir / f"extracted_tool_calls_round{round_id}_thinkgeo.json"
        output_file = base_dir / f"merged_tool_calls_round{round_id}_thinkgeo.json"
        if not input_file.exists():
            print(f"Skipping round {round_id}: {input_file} not found")
            continue
        with input_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        merged_data = []
        for question_data in data:
            merged_data.append(
                {
                    "question_index": question_data.get("question_index", ""),
                    "question_id": question_data.get("question_id", ""),
                    "tool_calls": merge_consecutive_same_tools(question_data.get("tool_calls", [])),
                }
            )
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"Round {round_id}: saved {len(merged_data)} questions -> {output_file}")


if __name__ == "__main__":
    main()
