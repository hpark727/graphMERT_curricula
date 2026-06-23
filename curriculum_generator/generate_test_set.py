"""
Generate the held-out test set: 3, 4, and 5-hop questions sampled from the
networking KG. Does NOT overlap with training triples by construction — test
questions require multi-hop compositional reasoning over paths not seen in SFT.

Output is ungraded. Run grade_curriculum_tinker.py on it afterward and keep
only GPT-OSS-passed questions as the final test benchmark.

Usage on della:
    python -m curriculum_generator.generate_test_set \
        --output   /scratch/gpfs/JHA/hp9084/curricula_gen/output/test_set_raw.json \
        --model_name google/gemma-3-27b-it \
        --hf_cache /scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface \
        --tensor_parallel_size 2 \
        --num_per_hop 3000 \
        --batch_size 64
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import random
from tqdm import tqdm

from curriculum_generator.generate_questions_local import (
    LocalLLMBackend, QAGeneratorLocal, _shuffle_answer_options
)


def generate_test_set(output_path, llm_backend, path_generator, num_per_hop,
                      batch_size, hop_counts=(3, 4, 5)):
    results = []
    next_id = 0
    hop_counts_done = {k: 0 for k in hop_counts}

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        next_id = max((r["id"] for r in results), default=-1) + 1
        for r in results:
            k = r["k_hops"]
            if k in hop_counts_done:
                hop_counts_done[k] += 1
        print(f"Resuming: {len(results)} already generated. Per-hop: {hop_counts_done}")

    # Build generation plan: how many more of each hop do we need
    todo = []
    for k in hop_counts:
        still_needed = num_per_hop - hop_counts_done[k]
        todo.extend([k] * still_needed)

    random.shuffle(todo)
    print(f"To generate: {len(todo)} questions ({num_per_hop} per hop for hops {hop_counts})")

    for batch_start in tqdm(range(0, len(todo), batch_size), desc="Generating test set"):
        batch_hops = todo[batch_start: batch_start + batch_size]

        # Sample paths for each hop count in the batch
        items = []
        for k in batch_hops:
            try:
                path = path_generator.generate_paths(k_hops=k)
                items.append({**path, "k_hops": k})
            except ValueError:
                items.append(None)

        valid_items = [it for it in items if it is not None]
        if not valid_items:
            continue

        # --- generate questions ---
        raw_questions = llm_backend.generate_questions_batch(valid_items)

        # --- parse + structural quality filter ---
        parsed = []
        for item, raw in zip(valid_items, raw_questions):
            try:
                question_text, answer = llm_backend.separate_question_and_answer(raw)
                question_text, answer = _shuffle_answer_options(question_text, answer)
                if not llm_backend.quality_filtering(question_text):
                    parsed.append(None)
                    continue
                parsed.append({**item, "question": question_text, "answer": answer})
            except Exception:
                parsed.append(None)

        # --- thinking traces ---
        valid_parsed = [(i, p) for i, p in enumerate(parsed) if p is not None]
        if valid_parsed:
            trace_inputs = [{"question": p["question"], "paths": p["paths"]}
                            for _, p in valid_parsed]
            traces = llm_backend.generate_thinking_traces_batch(trace_inputs)
            for (i, _), trace in zip(valid_parsed, traces):
                if trace:
                    parsed[i]["cot"] = trace

        # --- collect (no correctness filter — GPT grades everything) ---
        for p in parsed:
            if p is None or "cot" not in p:
                continue
            combined = llm_backend.combine_question_and_thinking_trace_with_answer(
                p["question"], p["cot"], p["answer"]
            )
            results.append({
                "id": next_id,
                "k_hops": p["k_hops"],
                "source_concept": p["source_concept"],
                "target_concept": p["target_concept"],
                "paths": p["paths"],
                "question_and_explanation": combined,
            })
            next_id += 1

        if (batch_start // batch_size) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            hop_summary = {k: sum(1 for r in results if r["k_hops"] == k)
                           for k in hop_counts}
            tqdm.write(f"[checkpoint] {len(results)} total | per hop: {hop_summary}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    hop_summary = {k: sum(1 for r in results if r["k_hops"] == k) for k in hop_counts}
    print(f"\nSaved {len(results)} ungraded test questions to {output_path}")
    print(f"Per-hop breakdown: {hop_summary}")
    print("Next: run grade_curriculum_tinker.py to get GPT-OSS-120b verdicts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",               default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/test_set_raw.json")
    parser.add_argument("--model_name",           default="google/gemma-3-27b-it")
    parser.add_argument("--hf_cache",             default="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size",           type=int, default=64)
    parser.add_argument("--num_per_hop",          type=int, default=3000,
                        help="Questions to generate per hop count (3000 → expect ~2300 after GPT grading)")
    parser.add_argument("--hop_counts",           type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    qa = QAGeneratorLocal(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_cache_dir=args.hf_cache,
    )

    generate_test_set(
        output_path=args.output,
        llm_backend=qa.llm,
        path_generator=qa.generator,
        num_per_hop=args.num_per_hop,
        batch_size=args.batch_size,
        hop_counts=tuple(args.hop_counts),
    )


if __name__ == "__main__":
    main()
