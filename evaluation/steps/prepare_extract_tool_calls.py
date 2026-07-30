#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adapted from the Earth-Agent evaluation framework
# (https://github.com/opendatalab/Earth-Agent), MIT License,
# Copyright (c) 2026 OpenDataLab. See the NOTICE file at the repo root.
import json
import re
import argparse
from pathlib import Path


def extract_tool_calls(log_file_path, output_file_path):
    """
    Extract tool calls from log file for each question
    
    Args:
        log_file_path: Input log file path  
        output_file_path: Output JSON file path
    """
    extracted_data = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by question records using regex to find question_index patterns
    import re
    # Find all positions where a new question starts (including error records)
    pattern_normal = r'\{\s*"question_index":\s*"(\d+)"'
    pattern_error = r'\{\s*"question_index":\s*\[\]'
    
    # Find all normal question records
    normal_matches = [(m.start(), m.group(1)) for m in re.finditer(pattern_normal, content)]
    # Find all error records  
    error_matches = [(m.start(), None) for m in re.finditer(pattern_error, content)]
    
    # Combine and sort all matches by position
    all_matches = normal_matches + error_matches
    all_matches.sort(key=lambda x: x[0])
    
    records = []
    for i, (start_pos, question_num) in enumerate(all_matches):
        if i + 1 < len(all_matches):
            end_pos = all_matches[i + 1][0]
            record_content = content[start_pos:end_pos].strip()
        else:
            record_content = content[start_pos:].strip()
        
        # Remove any trailing content after the last }
        # Find the position of the last }
        brace_count = 0
        end_idx = len(record_content)
        for j, char in enumerate(record_content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = j + 1
                    break
        
        if end_idx < len(record_content):
            record_content = record_content[:end_idx]
            
        records.append((record_content, question_num))
    
    for record_content, expected_question_num in records:
        try:
            # Parse the JSON record
            data = json.loads(record_content)
            question_index = data.get('question_index', '')
            
            # Handle error records with empty question_index
            if question_index == [] or question_index == "":
                conversations = data.get('conversations', '')
                # Extract question number from error message
                if isinstance(conversations, str) and 'Error processing question' in conversations:
                    match = re.search(r'Error processing question (\d+):', conversations)
                    if match:
                        question_index = match.group(1)
                        
                # Create error record
                if question_index:
                    question_data = {
                        "question_index": question_index,
                        "query": f"Error processing question {question_index}: {conversations}",
                        "tool_calls": [],
                        "error": True,
                        "timestamp": data.get('timestamp', '')
                    }
                    extracted_data.append(question_data)
                continue
                
            conversations = data.get('conversations', [])
            
            # Extract user query and tool calls
            query = ""
            tool_calls = []
            
            # Process conversations sequentially to maintain order
            for conv in conversations:
                if isinstance(conv, dict):
                    role = conv.get('role', '')
                    
                    # Extract user query
                    if role == 'user':
                        query = conv.get('content', '')
                    
                    # Extract assistant tool calls
                    elif role == 'assistant':
                        content = conv.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'name' in item and 'input' in item:
                                    # Store assistant call info temporarily
                                    tool_calls.append({
                                        'name': item.get('name', ''),
                                        'input': item.get('input', {}),
                                        'output': None  # Will be filled by corresponding tool response
                                    })
                    
                    # Extract tool outputs
                    elif role == 'tool':
                        tool_name = conv.get('name', '')
                        content = conv.get('content', [])
                        
                        # Find the corresponding assistant call
                        for tool_call in reversed(tool_calls):
                            if tool_call['name'] == tool_name and tool_call['output'] is None:
                                # Extract output text from tool response
                                if isinstance(content, list) and len(content) > 0:
                                    output_item = content[0]
                                    if isinstance(output_item, dict) and 'output' in output_item:
                                        output_list = output_item['output']
                                        if isinstance(output_list, list) and len(output_list) > 0:
                                            text_item = output_list[0]
                                            if isinstance(text_item, dict) and 'text' in text_item:
                                                tool_call['output'] = text_item['text']
                                break
            
            # Filter out tool calls that have both input and output
            completed_tool_calls = [tc for tc in tool_calls if tc['output'] is not None]
            
            # Create result data structure - include all questions even without tool calls
            if query:
                question_data = {
                    "question_index": question_index,
                    "query": query,
                    "tool_calls": completed_tool_calls  # Will be empty list if no tool calls
                }
                extracted_data.append(question_data)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing record: {e}")
            continue
    
    # Save extracted data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully extracted tool calls from {len(extracted_data)} questions")
    print(f"Results saved to: {output_file_path}")
    return extracted_data


def extract_tool_calls_all_occurrences(log_file_path):
    """
    Extract ALL occurrences of each question from log file
    
    Args:
        log_file_path: Input log file path
        
    Returns:
        dict: {question_id: [occurrence1, occurrence2, occurrence3, ...]}
    """
    question_occurrences = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all question records
    pattern_normal = r'\{\s*"question_index":\s*"(\d+)"'
    pattern_error = r'\{\s*"question_index":\s*\[\]'
    
    normal_matches = [(m.start(), m.group(1)) for m in re.finditer(pattern_normal, content)]
    error_matches = [(m.start(), None) for m in re.finditer(pattern_error, content)]
    
    all_matches = normal_matches + error_matches
    all_matches.sort(key=lambda x: x[0])
    
    records = []
    for i, (start_pos, question_num) in enumerate(all_matches):
        if i + 1 < len(all_matches):
            end_pos = all_matches[i + 1][0]
            record_content = content[start_pos:end_pos].strip()
        else:
            record_content = content[start_pos:].strip()
        
        # Remove trailing content after the last }
        brace_count = 0
        end_idx = len(record_content)
        for j, char in enumerate(record_content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = j + 1
                    break
        
        if end_idx < len(record_content):
            record_content = record_content[:end_idx]
            
        records.append((record_content, question_num))
    
    # Process each record
    for record_content, expected_question_num in records:
        try:
            data = json.loads(record_content)
            question_index = data.get('question_index', '')
            
            # Handle error records
            if question_index == [] or question_index == "":
                conversations = data.get('conversations', '')
                if isinstance(conversations, str) and 'Error processing question' in conversations:
                    match = re.search(r'Error processing question (\d+):', conversations)
                    if match:
                        question_index = match.group(1)
                        
                if question_index:
                    question_data = {
                        "question_index": question_index,
                        "query": f"Error processing question {question_index}: {conversations}",
                        "tool_calls": [],
                        "error": True,
                        "timestamp": data.get('timestamp', '')
                    }
                    if question_index not in question_occurrences:
                        question_occurrences[question_index] = []
                    question_occurrences[question_index].append(question_data)
                continue
            
            conversations = data.get('conversations', [])
            
            # Extract user query and tool calls
            query = ""
            tool_calls = []
            
            for conv in conversations:
                if isinstance(conv, dict):
                    role = conv.get('role', '')
                    
                    if role == 'user':
                        query = conv.get('content', '')
                    
                    elif role == 'assistant':
                        content = conv.get('content', [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and 'name' in item and 'input' in item:
                                    tool_calls.append({
                                        'name': item.get('name', ''),
                                        'input': item.get('input', {}),
                                        'output': None
                                    })
                    
                    elif role == 'tool':
                        tool_name = conv.get('name', '')
                        content = conv.get('content', [])
                        
                        for tool_call in reversed(tool_calls):
                            if tool_call['name'] == tool_name and tool_call['output'] is None:
                                if isinstance(content, list) and len(content) > 0:
                                    output_item = content[0]
                                    if isinstance(output_item, dict) and 'output' in output_item:
                                        output_list = output_item['output']
                                        if isinstance(output_list, list) and len(output_list) > 0:
                                            text_item = output_list[0]
                                            if isinstance(text_item, dict) and 'text' in text_item:
                                                tool_call['output'] = text_item['text']
                                break
            
            completed_tool_calls = [tc for tc in tool_calls if tc['output'] is not None]
            
            if query:
                question_data = {
                    "question_index": question_index,
                    "query": query,
                    "tool_calls": completed_tool_calls
                }
                
                if question_index not in question_occurrences:
                    question_occurrences[question_index] = []
                question_occurrences[question_index].append(question_data)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing record: {e}")
            continue
    
    return question_occurrences


def split_multi_attempt_results(log_file_path, summary_file_path, output_dir):
    """
    Split multi-attempt results into separate rounds based on attempt summaries
    
    Args:
        log_file_path: Input log file path
        summary_file_path: Path to results_summary.json containing attempt_summaries
        output_dir: Output directory for round-specific JSON files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load attempt summaries
    print(f"Loading attempt summaries from: {summary_file_path}")
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    attempt_summaries = summary_data.get('attempt_summaries', [])
    if not attempt_summaries:
        print("Error: No attempt_summaries found in summary file")
        return

    # Sort attempts to ensure deterministic processing
    attempt_summaries = sorted(attempt_summaries, key=lambda x: int(x.get('attempt', 0)))

    print(f"Found {len(attempt_summaries)} attempts")
    
    # Extract all occurrences of each question
    print(f"Extracting all occurrences from log file: {log_file_path}")
    question_occurrences = extract_tool_calls_all_occurrences(log_file_path)
    
    print(f"Extracted {len(question_occurrences)} unique questions")
    print(f"Total occurrences: {sum(len(v) for v in question_occurrences.values())}")
    
    # Get all question IDs (assumes attempts cover same question set)
    total_questions = int(attempt_summaries[0].get('questions_evaluated', 0))
    if total_questions <= 0:
        print("Error: invalid questions_evaluated in attempt summaries")
        return
    all_question_ids = [str(i) for i in range(1, total_questions + 1)]

    # Generic splitting rule:
    # For attempt N, each question should use the (1 + number_of_failures_in_attempts_<N) occurrence.
    # Because a question is re-run only when it failed in the previous attempt.
    failure_counts = {qid: 0 for qid in all_question_ids}  # how many times this qid failed so far

    round_results = {}
    for attempt in attempt_summaries:
        attempt_num = int(attempt.get('attempt', 0))
        if attempt_num <= 0:
            continue

        print(f"\n--- Processing Attempt {attempt_num} ---")
        print(f"Questions evaluated: {attempt.get('questions_evaluated')}")
        print(f"Failed count: {attempt.get('failed_count')}")

        round_data = []
        missing_in_log = 0
        fallback_used = 0

        for qid in all_question_ids:
            occurrences = question_occurrences.get(qid, [])
            if not occurrences:
                missing_in_log += 1
                continue

            occ_index = failure_counts.get(qid, 0)
            if occ_index < len(occurrences):
                round_data.append(occurrences[occ_index])
            else:
                # Log might be truncated or missing later occurrences; fall back to last available.
                fallback_used += 1
                round_data.append(occurrences[-1])

        round_results[attempt_num] = round_data

        output_file = output_path / f"extracted_tool_calls_round{attempt_num}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(round_data, f, ensure_ascii=False, indent=2)

        print(f"Round {attempt_num}: {len(round_data)} questions saved to {output_file}")
        if missing_in_log:
            print(f"  - Warning: {missing_in_log} questions missing in log, skipped")
        if fallback_used:
            print(f"  - Warning: {fallback_used} questions missing expected occurrence, used fallback")

        # Update failure counts AFTER producing current attempt.
        # Those failed in this attempt will have one more occurrence in the NEXT attempt.
        failed_ids = attempt.get('failed_ids', []) or []
        for fid in failed_ids:
            key = str(fid)
            if key in failure_counts:
                failure_counts[key] += 1
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    for attempt_num, data in round_results.items():
        print(f"Round {attempt_num}: {len(data)} questions")
    
    return round_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tool calls from model log file (multi-attempt splitting).")
    parser.add_argument("--base-dir", required=True, help="Base directory containing log + experiment_summary.json")
    parser.add_argument("--model-name", default="deepseek-chat", help="Used to build default log filename")
    parser.add_argument(
        "--rounds",
        default=None,
        help="Comma-separated round numbers to export, e.g. 1,2,3,4,5. Default: export all attempts in experiment_summary.json",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    log_file = f"{base_dir}/{args.model_name}_AP_langchain.log"
    summary_file = f"{base_dir}/experiment_summary.json"
    output_dir = base_dir

    print("="*60)
    print("Multi-Attempt Tool Call Extraction")
    print("="*60)
    print(f"Log file: {log_file}")
    print(f"Summary file: {summary_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Optionally filter attempt_summaries by rounds
    if args.rounds:
        rounds = {int(x.strip()) for x in str(args.rounds).split(",") if x.strip()}
        # Filter the selected attempt summaries before passing them to the splitter.
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        attempt_summaries = summary_data.get('attempt_summaries', []) or []
        attempt_summaries = sorted(attempt_summaries, key=lambda x: int(x.get('attempt', 0)))
        available = {int(a.get('attempt', 0)) for a in attempt_summaries if int(a.get('attempt', 0)) > 0}
        missing = sorted(rounds - available)
        if missing:
            print(f"Warning: requested rounds not found in experiment_summary.json, will skip: {missing}")
        summary_data['attempt_summaries'] = [a for a in attempt_summaries if int(a.get('attempt', 0)) in rounds]

        # The splitter accepts a file path, so persist the filtered summary beside
        # the other intermediate evaluation artifacts.
        tmp_summary = str(Path(output_dir) / "experiment_summary.filtered.json")
        with open(tmp_summary, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        split_multi_attempt_results(log_file, tmp_summary, output_dir)
    else:
        split_multi_attempt_results(log_file, summary_file, output_dir)
