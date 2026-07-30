#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the RSMeM evaluation suite.
import argparse
import json
import math
from typing import Dict, Tuple

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_total_tokens_single(data: Dict) -> int:
    """
    Extract total tokens from single run data
    
    Args:
        data: Question data dictionary
    
    Returns:
        total_tokens count
    """
    total_tokens = 0
    for question_id, question_data in data.items():
        if 'token_usage' in question_data:
            total_tokens += question_data['token_usage'].get('total_tokens', 0)
    return total_tokens

def extract_total_tokens(before_data: Dict, after_data: Dict) -> Tuple[int, int, int]:
    """
    Extract total tokens and reflection tokens
    
    Args:
        before_data: Data before reflection (attempt=1)
        after_data: Data after reflection (all attempts)
    
    Returns:
        (total_tokens_before, total_tokens_after, reflection_tokens)
    """
    total_tokens_before = 0
    total_tokens_after = 0
    reflection_tokens = 0
    
    # Calculate initial tokens for each question_id
    initial_tokens = {}
    for question_id, question_data in before_data.items():
        if 'token_usage' in question_data:
            token_count = question_data['token_usage'].get('total_tokens', 0)
            initial_tokens[question_id] = token_count
            total_tokens_before += token_count
    
    # Calculate total tokens after reflection and reflection tokens
    for question_id, question_data in after_data.items():
        if 'token_usage' in question_data:
            current_total = question_data['token_usage'].get('total_tokens', 0)
            total_tokens_after += current_total
            
            # Reflection token = current total - initial total
            # If attempt > 1, calculate reflection tokens
            attempt = question_data.get('attempt', 1)
            if attempt > 1 and question_id in initial_tokens:
                reflection_tokens += (current_total - initial_tokens[question_id])
    
    return total_tokens_before, total_tokens_after, reflection_tokens

def calculate_metrics(
    before_file: str,
    after_file: str,
    acc_before: float,
    acc_after: float
) -> Dict[str, float]:
    """
    Calculate Token Efficiency and RCBR metrics
    
    Args:
        before_file: JSON file path for before data
        after_file: JSON file path for after data
        acc_before: Accuracy before reflection (percentage)
        acc_after: Accuracy after reflection (percentage)
    
    Returns:
        Dictionary containing all metrics
    """
    # Load data
    before_data = load_json(before_file)
    after_data = load_json(after_file)
    
    # Calculate improvement gain
    G = acc_after - acc_before
    
    # Extract token usage
    total_tokens_before, total_tokens_after, reflection_tokens = extract_total_tokens(before_data, after_data)
    
    # Token Efficiency (η_before)
    if total_tokens_before > 0:
        eta_before = acc_before / math.log10(total_tokens_before)
    else:
        eta_before = 0.0
    
    # Token Efficiency (η_after)
    if total_tokens_after > 0:
        eta_after = acc_after / math.log10(total_tokens_after)
    else:
        eta_after = 0.0
    
    # Token Efficiency
    eta_improvement = eta_after - eta_before
    
    # RCBR
    # RCBR = G / (C_refl / C_total)
    if total_tokens_after > 0 and reflection_tokens > 0:
        reflection_ratio = reflection_tokens / total_tokens_after
        rcbr = G / reflection_ratio
    else:
        rcbr = 0.0
    
    return {
        'accuracy_before': acc_before,
        'accuracy_after': acc_after,
        'improvement_gain': G,
        'total_tokens_before': total_tokens_before,
        'total_tokens_after': total_tokens_after,
        'reflection_tokens': reflection_tokens,
        'reflection_ratio': reflection_tokens / total_tokens_after if total_tokens_after > 0 else 0.0,
        'token_efficiency_before': eta_before,
        'token_efficiency_after': eta_after,
        'token_efficiency_improvement': eta_improvement,
        'rcbr': rcbr
    }

def main():
    """CLI: compute token efficiency and RCBR from before/after aggregated results."""
    parser = argparse.ArgumentParser(
        description="Token efficiency (eta = Acc / log10(tokens)) and Reflection Cost-Benefit Ratio (RCBR).",
    )
    parser.add_argument("--before-file", required=True, help="results_final_aggregated.json before reflection (attempt=1)")
    parser.add_argument("--after-file", required=True, help="results_final_aggregated.json after all attempts")
    parser.add_argument("--acc-before", type=float, required=True, help="Accuracy before reflection, in [0, 1]")
    parser.add_argument("--acc-after", type=float, required=True, help="Accuracy after reflection, in [0, 1]")
    parser.add_argument("--output", default=None, help="Optional path to save the metrics as JSON")
    args = parser.parse_args()

    metrics = calculate_metrics(
        before_file=args.before_file,
        after_file=args.after_file,
        acc_before=args.acc_before,
        acc_after=args.acc_after,
    )

    print("=" * 70)
    print("Evaluation Metrics Results")
    print("=" * 70)
    print(f"Accuracy Before (Acc^(1)):              {metrics['accuracy_before']:.4f}")
    print(f"Accuracy After (Acc^(T)):               {metrics['accuracy_after']:.4f}")
    print(f"Improvement Gain (G):                   {metrics['improvement_gain']:.4f}")
    print("-" * 70)
    print(f"Total Tokens Before (C_before):         {metrics['total_tokens_before']:,}")
    print(f"Total Tokens After (C_after):           {metrics['total_tokens_after']:,}")
    print(f"Reflection Tokens (C_refl):             {metrics['reflection_tokens']:,}")
    print(f"Reflection Token Ratio:                 {metrics['reflection_ratio']:.2%}")
    print("-" * 70)
    print(f"Token Efficiency Before (eta_before):   {metrics['token_efficiency_before']:.6f}")
    print(f"Token Efficiency After (eta_after):     {metrics['token_efficiency_after']:.6f}")
    print(f"Token Efficiency Improvement (d_eta):   {metrics['token_efficiency_improvement']:.6f}")
    print("-" * 70)
    print(f"Reflection Cost-Benefit Ratio (RCBR):   {metrics['rcbr']:.4f}")
    print("=" * 70)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()