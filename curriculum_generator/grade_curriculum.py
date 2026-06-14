'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams


DOMAIN = "computer networking"


def _correctness_prompt(question_and_explanation: str, paths: list) -> str:
    paths_str = ', '.join([f"({p['start']}, {p['relation']}, {p['end']})" for p in paths])
    return f"""You are a senior {DOMAIN} engineer and exam question reviewer.
You are given a multiple-choice question, a reasoning explanation, and a selected answer.
The question was generated from this knowledge graph path: {paths_str}.

Evaluate the following:
1. Is the selected answer technically correct from a {DOMAIN} standpoint?
2. Does the reasoning explanation correctly justify that answer?
3. Are the three distractors plausibly wrong (not trick questions or obviously absurd)?

Respond in exactly this format — nothing else:
Correct: [Yes/No]

Question, explanation and answer:
{question_and_explanation}
"""


def _parse_verdict(text: str) -> str:
    match = re.search(r'Correct:\s*(Yes|No)', text, re.IGNORECASE)
    if match is None:
        return "error"
    return "Yes" if match.group(1).lower() == "yes" else "No"


def grade_curriculum(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    grader_name: str,
    tensor_parallel_size: int = 2,
    batch_size: int = 64,
    hf_cache_dir: str = None,
    max_model_len: int = 8192,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    verdict_path = os.path.join(output_dir, f"verdicts_{grader_name}.json")

    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions from {dataset_path}")

    verdicts = {}
    if os.path.exists(verdict_path):
        with open(verdict_path) as f:
            verdicts = json.load(f)
        print(f"Resuming: {len(verdicts)} verdicts already saved.")

    todo = [entry for entry in dataset if str(entry['id']) not in verdicts]
    print(f"To grade: {len(todo)} questions")

    if not todo:
        print("All questions already graded.")
        _print_stats(verdicts)
        return

    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype='bfloat16',
        download_dir=hf_cache_dir,
        max_model_len=max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    print("Model loaded.")

    is_qwen3 = "qwen3" in model_name.lower()

    def build_prompt(user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        if is_qwen3:
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                pass
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

    for i in tqdm(range(0, len(todo), batch_size), desc="Grading"):
        batch = todo[i:i + batch_size]
        prompts = [
            build_prompt(_correctness_prompt(entry['question_and_explanation'], entry['paths']))
            for entry in batch
        ]
        outputs = llm.generate(prompts, sampling_params)
        for entry, output in zip(batch, outputs):
            verdicts[str(entry['id'])] = _parse_verdict(output.outputs[0].text)

        if (i // batch_size + 1) % 5 == 0:
            with open(verdict_path, 'w') as f:
                json.dump(verdicts, f, indent=2)
            tqdm.write(f"[checkpoint] {len(verdicts)} verdicts saved.")

    with open(verdict_path, 'w') as f:
        json.dump(verdicts, f, indent=2)
    print(f"Done. Verdicts saved to {verdict_path}")
    _print_stats(verdicts)


def _print_stats(verdicts: dict) -> None:
    yes = sum(1 for v in verdicts.values() if v == "Yes")
    no = sum(1 for v in verdicts.values() if v == "No")
    err = sum(1 for v in verdicts.values() if v == "error")
    total = len(verdicts)
    print(f"Yes: {yes} ({100*yes/total:.1f}%)  No: {no} ({100*no/total:.1f}%)  Error: {err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to curriculum_dataset_hop_1_2.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write verdicts_<grader_name>.json")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--grader_name", type=str, required=True,
                        help="Short label for output file, e.g. qwen3 or gptoss")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--max_model_len", type=int, default=8192)
    args = parser.parse_args()

    grade_curriculum(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        grader_name=args.grader_name,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        hf_cache_dir=args.hf_cache_dir,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
