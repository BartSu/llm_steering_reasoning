
import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    if args.dataset == "minervamath":
        data_path = "data/minervamath/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = example["answer"]
                test_data.append({
                    "question": example["question"],
                    "gt": gt,
                })
    elif args.dataset == "olympiadbench":
        data_path = "data/olympiadbench/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = list(example["final_answer"])[0]
                test_data.append({
                    "question": example["question"],
                    "solution": list(example["solution"])[0],
                    "gt": gt,
                })
    elif args.dataset in ["aime24", "aime25"]:
        data_path = f"data/{args.dataset}/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                if args.dataset == "aime24":
                    gt = extract_box(example["solution"])
                elif args.dataset == "aime25":
                    gt = example["answer"]
                test_data.append({
                    "question": example["problem"],
                    "gt": gt,
                })
    elif args.dataset == "AMO-Bench":
        data_path = "data/AMO-Bench/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["answer"])
                test_data.append({
                    "question": example["prompt"],
                    "answer": example["solution"],
                    "gt": gt,
                })
    elif args.dataset == "MATH500":
        data_path = "data/MATH500/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt,
                })
    elif args.dataset in ["gsm8k_test", "gsm8k_train", "gsm8k-platinum_test"]:
        if args.dataset == "gsm8k_train":
            data_path = "data/gsm8k/train.jsonl"
        elif args.dataset == "gsm8k_test":
            data_path = "data/gsm8k/test.jsonl"
        elif args.dataset == "gsm8k-platinum_test":
            data_path = "data/gsm8k-platinum/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError("Dataset not supported")
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"

    # optionally steer the model with an emotion-specific system prompt
    if hasattr(args, "emotion") and args.emotion is not None and args.emotion != "":
        prefix += f"You should respond while exhibiting a {args.emotion} emotional tone.\n"

    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if args.use_chat_format:
            if "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path:
                # models without an explicit system role: fold system/emotion instructions into the user message
                messages = [{
                    "role": "user",
                    "content": prefix + "Question: " + example["question"].strip()
                }]
            else:
                # standard chat models: use a dedicated system prompt that encodes emotion steering
                messages = [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Question: " + example["question"].strip()},
                ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                swap_space=16,
                gpu_memory_utilization=0.95, 
                tensor_parallel_size=torch.cuda.device_count(), 
                # enforce_eager=True, # set False to allow speed up
                max_model_len=args.max_tokens + 2000
            )


    sampling_params = SamplingParams(n=args.num_samples,
                                    temperature=args.temperature,
                                    max_tokens=args.max_tokens,
                                    skip_special_tokens=False)


    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)
    

    outputs = [[trim_output(o) for o in output] for output in result]

    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution": example["answer"] if "answer" in example else None,
        "model_generation": output,
    } for example, output, prompt in zip(test_data, outputs, prompts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm8k_test"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k_test",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="Optional emotional tone to encode in the system prompt (e.g., 'happy', 'sad', 'confident').",
    )
    args = parser.parse_args()
    
    main(args)