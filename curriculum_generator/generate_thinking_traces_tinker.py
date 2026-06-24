"""
Stage 2: Read Gemma-generated questions (question + answer, no thinking trace),
generate thinking traces with GPT-OSS-120b via Tinker, run correctness grading,
and save passing entries as a fully-formed dataset ready for tokenization.

Runs locally against the Tinker API with 16 concurrent workers.

Usage:
    python -m curriculum_generator.generate_thinking_traces_tinker \
        --input   curriculum_generator/data/curriculum_dataset_gap_fill_redundant.json \
        --output  curriculum_generator/data/curriculum_dataset_gap_fill_redundant_final.json \
        --concurrency 16

    python -m curriculum_generator.generate_thinking_traces_tinker \
        --input   curriculum_generator/data/test_set_raw.json \
        --output  curriculum_generator/data/test_set_final.json \
        --concurrency 16
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import json
import re
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from curriculum_generator.generate_questions_tinker import TinkerLLMBackend


def process_one(entry, backend):
    """Generate thinking trace and grade a single question entry."""
    paths = entry["paths"]
    question = entry["question"]
    answer = entry["answer"]

    # Generate thinking trace
    cot = backend.generate_thinking_trace(question=question, paths=paths)
    if not cot:
        return None

    combined = backend.combine_question_and_thinking_trace_with_answer(
        question, cot, answer
    )

    # Correctness filter
    verdict = backend.correctness_filtering(combined, paths)
    if verdict != "Yes":
        return None

    return {
        "id": entry["id"],
        "k_hops": entry["k_hops"],
        "source_concept": entry["source_concept"],
        "target_concept": entry["target_concept"],
        "paths": paths,
        "question_and_explanation": combined,
    }


def run(input_path, output_path, concurrency):
    with open(input_path) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} question-only entries from {input_path}")

    # Resume from partial output
    results = []
    done_ids = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(results)} already done.")

    todo = [e for e in dataset if e["id"] not in done_ids]
    print(f"To process: {len(todo)} (concurrency={concurrency})")

    if not todo:
        print("Nothing to do.")
        _print_stats(dataset, results)
        return results

    _local = threading.local()

    def get_backend():
        if not hasattr(_local, "llm"):
            _local.llm = TinkerLLMBackend(model_name="openai/gpt-oss-120b")
        return _local.llm

    lock = threading.Lock()
    checkpoint_counter = [0]
    failed = [0]

    def worker(entry):
        result = None
        try:
            result = process_one(entry, get_backend())
        except Exception as e:
            tqdm.write(f"[error] id={entry['id']}: {e}")

        with lock:
            if result is not None:
                results.append(result)
            else:
                failed[0] += 1
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % 100 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                tqdm.write(f"[checkpoint] {len(results)} passed, {failed[0]} failed")
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(worker, e): e for e in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Thinking traces"):
            future.result()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} passed, {failed[0]} failed.")
    print(f"Saved to {output_path}")
    _print_stats(dataset, results)
    return results


def _print_stats(dataset, results):
    from collections import Counter
    hop_counts = Counter(r["k_hops"] for r in results)
    total = len(dataset)
    passed = len(results)
    print(f"Pass rate: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Per-hop breakdown: {dict(sorted(hop_counts.items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True,
                        help="Questions-only JSON from Gemma generation stage")
    parser.add_argument("--output",      required=True,
                        help="Output path for fully-formed Q&A with thinking traces")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    run(args.input, args.output, args.concurrency)


if __name__ == "__main__":
    main()
