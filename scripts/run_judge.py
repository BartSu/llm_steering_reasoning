import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial

import sys
import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")

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

def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans
    
 
def main(args):
    random.seed(42)

    print("Loading data...")

    with open(args.predictions_path) as fin:
        predictions = [json.loads(line) for line in fin]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path
    )

    # Set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # Set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare prompts based on dataset type
    prompts = []
    for prediction in predictions:
        prefix = "Please evaluate the following.\n"
        postfix = """Act as a rigorous verifier. Your output must follow this structured checklist exactly (do **not** expose chain-of-thought — give concise, externally usable statements only):

        1. One-line summary
        - Restate the Question's final claim in a single sentence.

        2. Step-by-step verification
        For each numbered step or paragraph in the "Proposed Reasoning", produce a short entry with this sub-structure:
            a. Quoted step (≤2 lines) — copy the exact sentence or formula you evaluate.
            b. Verdict — write either "Correct" or "Incorrect".
            c. Concise justification (1–2 short bullets):
                - If the step is incorrect, identify *why* (logical gap, false assumption, misuse of definition, omitted case, algebraic error, incorrect inference, invalid generalization, etc.).
                - For any arithmetic or algebra, show the calculation **digit-by-digit** to verify — do the math explicitly and briefly.
            d. Severity — label as "Minor", "Major", or "Critical".
            e. Impact on conclusion — state "Affects final answer" or "Does not affect final answer".

        3. Fallacies & errors (concise bullets)
        - Produce a short bullet list of the distinct reasoning fallacies or mistakes you found. For each bullet include:
            - a short label (e.g., "False premise", "Division by zero risk", "Unjustified generalization", "Off-by-one arithmetic error"),
            - the exact step number or quoted phrase where it appears,
            - one-line effect on validity.

        4. Minimal correction (if incorrect)
        - Provide a compact corrected step or short alternative argument (≤3 sentences) that removes the identified error(s). If correction is impossible without new assumptions, say so and state which additional facts are required.

        5. Final verdict (strict)
        - End with a single-line boxed result indicating whether the entire Proposed Reasoning is accurate:
            - If the reasoning is fully correct (no critical errors that change the outcome), write \\boxed{1}.
            - Otherwise write \\boxed{0}.
        - The box must be placed at the end of the line.

        Formatting rules:
        - Use numbered steps matching the original Proposed Reasoning steps when possible.
        - Keep each justification concise; prefer bullets and short sentences.
        - **Do not** provide internal chain-of-thought. Provide only the external checks, math work, and concise corrections above.
        - Always check arithmetic digit-by-digit (no mental shortcuts).
        """

        prompt = (
            prefix
            + "**[Question]**\n"
            + prediction["problem"].strip()
            + "\n**[Proposed Reasoning & Solution]**\n"
            + prediction["model_generation"][0].strip()
            + "\n**[Your Task]**\n"
            + postfix
        )

        if args.use_chat_format:
            if "gemma" in args.model_name_or_path.lower() or "deepseek" in args.model_name_or_path.lower():
                messages = [{"role": "user", "content": prefix + "Question: " + prediction["problem"].strip() + "\nResponse: " + prediction["model_generation"][0].strip()}]
            else:
                messages = [
                    {"role": "system", "content": prefix}, 
                    {"role": "user", "content": prompt}
                ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, reasoning_effort="high")
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        
        prompts.append(prompt)
    
    # Save example prompt
    with open(os.path.join(args.save_dir, "example_judge_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print("Loading model...")
    model = LLM(
        model=args.model_name_or_path, 
        tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, 
        swap_space=16, 
        gpu_memory_utilization=0.95, 
        tensor_parallel_size=torch.cuda.device_count(), 
        max_model_len=args.max_tokens + 16384
    )

    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=0.7,
        max_tokens=args.max_tokens
    )

    print("Generating judgements...")

    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        result.append(attempts)

    outputs = [[o for o in output] for output in result]

    judgements = [{
        "problem": prediction["problem"],
        "model_generation": output
    } for prediction, output in zip(predictions, outputs)]

    # Save judgements
    with open(os.path.join(args.save_dir, "judgements.jsonl"), "w") as fout:
        for judgement in judgements:
            fout.write(json.dumps(judgement) + "\n")

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
        default=32768,
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
        "--predictions_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
