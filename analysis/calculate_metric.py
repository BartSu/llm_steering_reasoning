#!/usr/bin/env python3
"""
Calculate metrics from predictions.jsonl file.
Based on the analysis notebook, this script computes:
- accuracy: overall accuracy using first generation
- avg_tokens: average number of tokens per generation
- passK: pass@K metrics for all K values
- true_n_ls: matrix of correctness for each generation
"""

import json
import os
import sys
import argparse
from transformers import AutoTokenizer
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
import numpy as np


def compile_predictions(predictions_file, model_name):
    """
    Calculate accuracy and average tokens from predictions file.
    
    Args:
        predictions_file: Path to predictions.jsonl file
        model_name: Name of the model for tokenization
        
    Returns:
        accuracy: Overall accuracy (using first generation)
        avg_tokens: Average number of tokens per generation
    """
    extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    total_tokens = 0
    with open(predictions_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            gold = parse(f"${data['answer']}$", extraction_config=extraction_target)
            
            # Get model generation output (usually first item)
            llm_output = data['model_generation'][0] if isinstance(data['model_generation'], list) else data['model_generation']
            answer = parse(llm_output, extraction_config=extraction_target)
            total_tokens += len(tokenizer.encode(llm_output))
            result = verify(gold, answer)
            results.append(result)
    
    accuracy = sum(results) / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    return accuracy, avg_tokens


def compile_predictions_passK(predictions_file, model_name, total_questions_num, n, seed_max):
    """
    Calculate pass@K metrics from predictions file.
    
    Args:
        predictions_file: Path to predictions.jsonl file
        model_name: Name of the model (not used but kept for compatibility)
        total_questions_num: Total number of questions
        n: Number of generations per question
        seed_max: Number of seeds (usually 1)
        
    Returns:
        pass_at_k_mean_ls: Array of pass@K values for K=1 to N
        True_n_ls: Matrix of correctness (N x total_questions_num)
    """
    extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    
    # jsonl file, each line is a json object
    # each json object has a key 'model_generation', the number of generations is N
    N = n * seed_max
    
    def read_scores_from_jsonl(file_path):
        try:
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
            
            scores = []
            for item in data:
                gold = parse(f"${item['answer']}$", extraction_config=extraction_target)
                item_scores = []
                for generation in item['model_generation']:
                    answer = parse(generation, extraction_config=extraction_target)
                    result = verify(gold, answer)
                    item_scores.append(result)
                scores.append(item_scores)
            return scores
        except Exception as e:
            print(f"Error reading file {file_path}: {e}", file=sys.stderr)
        return []
    
    def read_jsonl_file(file_path, model_name, True_n_ls, total_questions_num, N, seed_max):
        for seed in range(1, seed_max + 1):
            scores = read_scores_from_jsonl(file_path)
            for i in range(total_questions_num):
                for j in range(n):
                    True_n_ls[(seed-1)*n+j][i] = int(scores[i][j])
            
    def pass_at_k(n, c, k):
        if n - c < k: 
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    True_n_ls = [[0 for i in range(total_questions_num)] for j in range(N)]
    read_jsonl_file(predictions_file, model_name, True_n_ls, total_questions_num, N, seed_max)
    True_n_ls = np.array(True_n_ls)
    
    correct_ls = [0 for i in range(total_questions_num)]
    for i in range(total_questions_num):
        for j in range(n * seed_max):
            correct_ls[i] += True_n_ls[j][i]    
    
    pass_at_k_ls = [[] for i in range(N)]
    for i in range(N):
        for j in range(total_questions_num):
            pass_at_k_ls[i].append(pass_at_k(n * seed_max, correct_ls[j], i+1))
    
    pass_at_k_mean_ls = [np.mean(pass_at_k_ls[i]) for i in range(N)]
    
    pass_at_k_mean_ls = np.array(pass_at_k_mean_ls)
            
    return pass_at_k_mean_ls, True_n_ls


def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from predictions.jsonl file')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to predictions.jsonl file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for tokenization (e.g., Qwen/Qwen3-4B)')
    parser.add_argument('--output', type=str, default=None, help='Output metrics.json path (default: same directory as predictions_file)')
    parser.add_argument('--seed_max', type=int, default=1, help='Number of seeds (default: 1)')
    
    args = parser.parse_args()
    
    predictions_file = args.predictions_file
    if not os.path.exists(predictions_file):
        print(f"Error: predictions file not found: {predictions_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        metric_path = args.output
    else:
        metric_path = os.path.join(os.path.dirname(predictions_file), "metrics.json")
    
    # Skip if metrics.json already exists
    if os.path.exists(metric_path):
        print(f"Metrics file already exists: {metric_path}. Skipping...", file=sys.stderr)
        sys.exit(0)
    
    print(f"Calculating metrics for {predictions_file}...", file=sys.stderr)
    
    # Calculate accuracy and avg_tokens
    accuracy, avg_tokens = compile_predictions(predictions_file, args.model_name)
    
    # Get number of questions and generations
    with open(predictions_file, 'r') as f:
        total_questions_num = sum(1 for line in f)
    
    with open(predictions_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            n = len(data['model_generation'])
            break
    
    # Calculate pass@K metrics
    passK, True_n_ls = compile_predictions_passK(
        predictions_file, 
        args.model_name, 
        total_questions_num, 
        n, 
        args.seed_max
    )
    
    # Format pass@K values
    passK_dict = {}
    for i in range(len(passK)):
        passK_dict[f"pass{i+1}"] = f"{passK[i]:.4f}"
    
    # Create output dictionary
    json_output = {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "passK": passK_dict,
        "true_n_ls": True_n_ls.tolist()
    }
    
    # Write to file
    with open(metric_path, 'w') as f:
        json.dump(json_output, f)
    
    print(f"Metrics saved to {metric_path}", file=sys.stderr)
    print(f"Accuracy: {accuracy:.4f}, Avg Tokens: {avg_tokens:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
