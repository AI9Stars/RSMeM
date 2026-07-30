#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adapted from the Earth-Agent evaluation framework
# (https://github.com/opendatalab/Earth-Agent), MIT License,
# Copyright (c) 2026 OpenDataLab. See the NOTICE file at the repo root.
from __future__ import annotations

import json
import os
import sys
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- Configuration ---
# Credentials are resolved from CLI flags (--api-key/--base-url), which the
# evaluation pipeline forwards from your configs/*.yaml. The environment
# variables RSMeM_API_KEY / RSMeM_BASE_URL act as a fallback for standalone runs.
def resolve_credentials(cli_api_key: str = "", cli_base_url: str = "") -> tuple[str, str]:
    api_key = cli_api_key or os.getenv("RSMeM_API_KEY", "")
    base_url = cli_base_url or os.getenv("RSMeM_BASE_URL", "")
    return api_key, base_url


def create_client(api_key: str, base_url: str):
    if not api_key or not base_url:
        raise RuntimeError(
            "Missing API configuration. Pass --api-key/--base-url (or set "
            "RSMeM_API_KEY / RSMeM_BASE_URL) so answers can be polished."
        )
    from openai import OpenAI

    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        raise


def gpt4o_polish_answer(question: str, answer: str, client: OpenAI, model: str) -> str:
    """
    Use GPT-4o to extract the final answer from the model's response 
    using the official openai SDK.
    """
    # Construct prompt for GPT-4o
#     prompt = f"""You are tasked with extracting the final answer from a model's response to a multiple-choice question.
# The response may contain a <think> section which you should ignore for the final answer extraction, but it might help you understand the model's reasoning.

# Question:
# {question}

# Model's Response:
# {answer}

# Please extract ONLY the final answer from the model's response. The answer should be in the format <Answer>X<Answer> where X is the choice (A, B, C, or D).
# If the model provided multiple answers or explanations, extract only the final choice.
# If the answer contains something like "D.8.5 K higher", just extract "D".

# Return ONLY the answer in this exact format: <Answer>X<Answer>
# Do not include any explanation or additional text.
# Attention:
# The most important thing is:
# You just need to extract the normal answer, if the "answer" has some errors, you just need to keep it and don't give me a answer determined by yourself! For example, the model_answer is Sorry, need more steps to process this request. you just need to keep this, don't give me an answer you guess!!!"""

    prompt = f"""You are an expert at extracting final answers from agent logs. 
Your goal is to identify the final multiple-choice selection (A, B, C, or D) from the provided response.

### Rules:
1. **Strict Extraction**: Only extract the answer if the agent explicitly reached a conclusion.
2. **No Guessing**: If the response is truncated (ends mid-sentence), contains only an error message (e.g., "Error code: 400"), or says "I need more steps", do NOT output the <Answer>X<Answer> format.
3. **Fallback**: If no clear answer is found, return the string "No valid answer found".
4. **Format**: If a clear answer exists, return ONLY <Answer>X<Answer>.

### Examples:
- Input: "<Answer>D<Answer>" -> Output: <Answer>D<Answer>
- Input: "<Answer>D.355.82<Answer>" -> Output: <Answer>D<Answer>
- Input: "Based on the analysis, the answer is A." -> Output: <Answer>A<Answer>
- Input: "The trend is increasing, so D. 8.5K" -> Output: <Answer>D<Answer>
- Input: "I will first calculate the mean and then..." (truncated) -> Output: No valid answer found
- Input: "Sorry, need more steps to process this request." -> Output: No valid answer found

### Question:
{question}

### Model's Response:
{answer}

### Final Output:""" 
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts final answers from model responses."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=5000,
            timeout=30
        )
        
        result = response.choices[0].message.content.strip()
        if not result:
            raise ValueError("Empty response from API")
        # Prefer the canonical tagged answer returned by the polishing model.
        match = re.search(r'<Answer>([A-D])[^<]*<Answer>', result)
        if match:
            return f"<Answer>{match.group(1)}<Answer>"
        
        # Otherwise accept a standalone multiple-choice letter.
        match = re.search(r'\b([A-D])\b', result)
        if match:
            return f"<Answer>{match.group(1)}<Answer>"
        
        return result
            
    except Exception as e:
        print(f"Error calling API: {e}")
        # Fall back to extracting an answer from the unpolished response.
        match = re.search(r'<Answer>([^<]+)<Answer>', answer)
        if match:
            letter_match = re.search(r'([A-D])', match.group(1))
            if letter_match:
                return f"<Answer>{letter_match.group(1)}<Answer>"
        return answer


def filter_existing_polished_rounds(base_dir: str, rounds: list) -> list:
    """Drop any round whose polished output already exists in *base_dir*.

    Generalises the old ROUND1-only special case: for every requested round
    ``r``, if ``results_summary_polished_ROUND{r}.jsonl`` is already present,
    that round is skipped. This matters for resume-through-attempt runs whose
    early rounds are byte-identical to the source and whose polished answers
    are carried over by ``rsmem.io.seed_resume_files`` — re-polishing them would
    only waste LLM calls. Order is preserved.
    """
    kept = []
    for r in rounds:
        polished = os.path.join(base_dir, f"results_summary_polished_ROUND{r}.jsonl")
        if os.path.exists(polished):
            print(f"Detected existing polished ROUND{r}, skipping.")
        else:
            kept.append(r)
    return kept


def load_log_file(log_path: Path) -> dict:
    """
    Load and parse the log file to extract questions
    """
    questions = {}

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            buffer = ""
            for line in f:
                buffer += line
                try:
                    data = json.loads(buffer) 
                except json.JSONDecodeError:
                    continue  
                except Exception as e:
                    print(f"Error processing JSON line: {e}")
                    buffer = ""
                    continue
                else:
                    buffer = ""
                    question_id = data.get('question_index', '')
                    if isinstance(question_id, list):
                        question_id = str(question_id[0]) if question_id else ''
                    elif not isinstance(question_id, str):
                        question_id = str(question_id)
                    
                    if question_id.startswith('question'):
                        question_id = question_id[8:]

                    conversations = data.get('conversations', [])
                    for conv in conversations:
                        if isinstance(conv, dict) and conv.get('role') == 'user':
                            user_content = conv.get('content', '')
                            questions[question_id] = user_content
                            break
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        sys.exit(1)
    
    return questions


def polish_jsonl_results(input_file: str, output_file: str, log_file: str, client: OpenAI, model: str):
    """
    Polish the JSONL results file using GPT-4o
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    log_path = Path(log_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading questions from log file: {log_file}")
    questions = load_log_file(log_path)
    print(f"Loaded {len(questions)} questions")
    
    print(f"Processing {input_file} -> {output_file}")
    
    total_processed = 0
    success_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
                question_id = str(item.get('question_id', ''))
                answer = item.get('answer', '')
                
                question = questions.get(question_id, '')
                
                if not question:
                    print(f"Warning: No question found for ID {question_id}, skipping polishing.")
                    polished_answer = answer
                else:
                    total_processed += 1
                    print(f"Processing {input_path} question {question_id}...")
                    #  gpt4o_polish_answer 
                    if "Error code:" in answer or "Connection error" in answer:
                        polished_answer = "Error in original response"
                    else:
                        polished_answer = gpt4o_polish_answer(question, answer, client, model)
                
                # Create polished item
                polished_item = item.copy()
                polished_item["original_answer"] = answer
                polished_item["final_answer"] = polished_answer
                
                # Write to output JSONL
                f_out.write(json.dumps(polished_item, ensure_ascii=False) + '\n')
                
                if re.search(r'<Answer>([A-D])<Answer>', polished_answer):
                    success_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {e}")
                continue

    print(f"\nFinished! Processed {total_processed} items.")
    print(f"Successfully extracted answers: {success_count}/{total_processed}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polish jsonl answers by extracting final choice with a GPT model.")
    parser.add_argument("--base-dir", required=True, help="Directory containing results_summary_ROUND{N}.jsonl")
    parser.add_argument(
        "--model-name",
        default="deepseek-chat",
        help="AGENT model that produced the run being evaluated (e.g. glm5.2). Used ONLY to "
             "locate the log file {model-name}_AP_langchain.log for the question text. "
             "This is NOT the polish/judge LLM (see --model).",
    )
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="Comma-separated round numbers, e.g. 1,2,3",
    )
    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent rounds to process")
    parser.add_argument("--api-key", default="", help="LLM API key (falls back to env RSMeM_API_KEY)")
    parser.add_argument("--base-url", default="", help="LLM base URL (falls back to env RSMeM_BASE_URL)")
    parser.add_argument(
        "--model",
        default="",
        help="POLISH/JUDGE LLM that extracts the final <Answer>X<Answer> from each agent "
             "response (e.g. deepseek-v4-flash). Distinct from --model-name. Normally forwarded "
             "by run_pipeline from the --config yaml's `model:`. Falls back to env "
             "RSMeM_POLISH_MODEL. There is no hardcoded default so a missing/unreachable judge "
             "fails loudly instead of silently mislabeling every answer UNKNOWN.",
    )
    args = parser.parse_args()


    base_dir = args.base_dir
    log_file = f"{base_dir}/{args.model_name}_AP_langchain.log"
    rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]
    # --- Skip ANY round whose polished output already exists ---
    rounds = filter_existing_polished_rounds(base_dir, rounds)

    if not rounds:
        print("No rounds to process after skipping existing outputs.")
        sys.exit(0)
        
    # Resolve the polish/judge model from CLI (forwarded by run_pipeline from the
    # --config yaml) or env, with NO hardcoded fallback: a missing judge model must
    # fail loudly rather than silently defaulting to an unreachable model (which would
    # 401 on every call and mislabel all answers UNKNOWN).
    polish_model = args.model or os.getenv("RSMeM_POLISH_MODEL", "")
    if not polish_model:
        print(
            "Error: no polish/judge model configured. Set `model:` in your --config yaml "
            "(run_pipeline forwards it as --model), pass --model directly, or set "
            "RSMeM_POLISH_MODEL.",
            file=sys.stderr,
        )
        sys.exit(2)
    api_key, base_url = resolve_credentials(args.api_key, args.base_url)

    def run_one_round(r: int) -> str:
        input_file = os.path.join(base_dir, f"results_summary_ROUND{r}.jsonl")
        output_file = os.path.join(base_dir, f"results_summary_polished_ROUND{r}.jsonl")

        if not os.path.exists(input_file):
            return f"Skipping Round {r}: {input_file} not found."

        client = create_client(api_key, base_url)
        polish_jsonl_results(input_file, output_file, log_file, client, model=polish_model)
        return f"Round {r} finished: {output_file}"

    max_workers = max(1, min(int(args.max_workers), len(rounds)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_round, r): r for r in rounds}
        for future in as_completed(futures):
            r = futures[future]
            try:
                msg = future.result()
                print(msg)
            except Exception as e:
                print(f"Round {r} failed: {e}")
