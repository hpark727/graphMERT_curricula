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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from curriculum_generator.generate_questions_tinker import TinkerLLMBackend


def grade_curriculum_tinker(
    dataset_path: str,
    output_dir: str,
    model_name: str,
    grader_name: str,
    concurrency: int = 16,
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
    print(f"To grade: {len(todo)} questions  (concurrency={concurrency})")

    if not todo:
        print("All questions already graded.")
        _print_stats(verdicts)
        return

    # thread-local storage so each thread gets its own backend + async event loop
    _local = threading.local()

    def get_backend():
        if not hasattr(_local, 'llm'):
            _local.llm = TinkerLLMBackend(model_name=model_name)
        return _local.llm

    lock = threading.Lock()
    checkpoint_counter = [0]

    def grade_one(entry):
        try:
            backend = get_backend()
            paths = entry['paths']
            paths_str = ', '.join(
                [f"({p['start']}, {p['relation']}, {p['end']})" for p in paths]
            )
            prompt = f"""You are a senior computer networking engineer and exam question reviewer.
You are given a multiple-choice question, a reasoning explanation, and a selected answer.
The question was generated from this knowledge graph path: {paths_str}.

Evaluate the following:
1. Is the selected answer technically correct from a computer networking standpoint?
2. Does the reasoning explanation correctly justify that answer?
3. Are the three distractors plausibly wrong (not trick questions or obviously absurd)?

Respond in exactly this format — nothing else:
Correct: [Yes/No]

Question, explanation and answer:
{entry['question_and_explanation']}"""
            # 2048 tokens covers gpt-oss-120b's analysis scratchpad + final answer
            content = backend._generate(prompt, max_tokens=2048, temperature=0.0)
            match = re.search(r'Correct:\s*(Yes|No)', content, re.IGNORECASE)
            verdict = ("Yes" if match.group(1).lower() == "yes" else "No") if match else "error"
        except Exception as e:
            tqdm.write(f"[error] id={entry['id']}: {e}")
            verdict = "error"
        with lock:
            verdicts[str(entry['id'])] = verdict
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % 100 == 0:
                with open(verdict_path, 'w') as f:
                    json.dump(verdicts, f, indent=2)
        return entry['id'], verdict

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(grade_one, entry): entry for entry in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Grading"):
            future.result()

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
    parser.add_argument("--dataset", required=True,
                        help="Path to curriculum_dataset_hop_1_2.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--grader_name", type=str, default="gptoss120b",
                        help="Short label for output file")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Number of parallel API calls")
    args = parser.parse_args()

    grade_curriculum_tinker(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model_name=args.model_name,
        grader_name=args.grader_name,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
