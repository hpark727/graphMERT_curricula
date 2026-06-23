"""
Stage 1 of coverage gap-fill: find uncovered KG triples and generate targeted
1-hop Q&A pairs using Gemma 27B on della (same generator as the original pipeline).

Output is an ungraded JSON file with the same format as curriculum_dataset_hop_3.json.
Run grade_curriculum_tinker.py on it afterward to get GPT-OSS-120b verdicts.

Usage on della (after sbatch or interactive session with GPU):
    python -m curriculum_generator.fill_coverage_gaps \
        --full_dataset  /scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_hop_3.json \
        --verdicts      /scratch/gpfs/JHA/hp9084/curricula_gen/output/verdicts_gptoss120b.json \
        --kg_triples    /scratch/gpfs/JHA/hp9084/curricula_gen/networks_kg/validated_triples.csv \
        --output        /scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_gap_fill.json \
        --model_name    google/gemma-3-27b-it \
        --hf_cache      /scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface \
        --tensor_parallel_size 2 \
        --batch_size    32 \
        --max_retries   5
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import argparse
import re
import random
from tqdm import tqdm

from curriculum_generator.generate_questions_local import LocalLLMBackend, _shuffle_answer_options


def find_uncovered_triples(full_dataset_path, verdicts_path, kg_triples_path):
    with open(full_dataset_path) as f:
        dataset = json.load(f)
    with open(verdicts_path) as f:
        verdicts = json.load(f)

    covered = set()
    for idx_str, verdict in verdicts.items():
        if verdict == "Yes":
            for hop in dataset[int(idx_str)]["paths"]:
                covered.add((hop["start"], hop["relation"], hop["end"]))

    all_triples = []
    with open(kg_triples_path) as f:
        for row in csv.DictReader(f):
            all_triples.append((row["head"].strip(), row["relation"].strip(), row["tail"].strip()))

    uncovered = [t for t in all_triples if t not in covered]

    print(f"KG triples total   : {len(all_triples)}")
    print(f"Covered by GPT-yes : {len(covered & set(all_triples))} "
          f"({100 * len(covered & set(all_triples)) / len(all_triples):.1f}%)")
    print(f"Uncovered          : {len(uncovered)}")
    return uncovered


def generate_for_triples(uncovered, output_path, llm, batch_size, max_retries):
    # Resume from partial output
    results = []
    done_triples = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        for r in results:
            p = r["paths"][0]
            done_triples.add((p["start"], p["relation"], p["end"]))
        print(f"Resuming: {len(results)} already generated.")

    todo = [t for t in uncovered if t not in done_triples]
    print(f"To generate: {len(todo)} triples")

    if not todo:
        print("Nothing to do.")
        return results

    next_id = max((r["id"] for r in results), default=-1) + 1

    # Process in batches; retry failed triples up to max_retries times
    remaining = list(todo)
    for retry in range(max_retries):
        if not remaining:
            break

        print(f"\n--- Pass {retry + 1}/{max_retries}: {len(remaining)} triples to attempt ---")
        failed_this_pass = []

        for batch_start in tqdm(range(0, len(remaining), batch_size), desc=f"Pass {retry+1}"):
            batch = remaining[batch_start: batch_start + batch_size]

            # Build path items for this batch
            items = [
                {
                    "source_concept": head,
                    "target_concept": tail,
                    "paths": [{"start": head, "relation": rel, "end": tail}],
                }
                for head, rel, tail in batch
            ]

            # --- generate questions (one vLLM batch call) ---
            raw_questions = llm.generate_questions_batch(items)

            # --- parse answers and quality-filter ---
            parsed = []
            for item, raw in zip(items, raw_questions):
                try:
                    question_text, answer = llm.separate_question_and_answer(raw)
                    question_text, answer = _shuffle_answer_options(question_text, answer)
                    if not llm.quality_filtering(question_text):
                        parsed.append(None)
                        continue
                    parsed.append({**item, "question": question_text, "answer": answer})
                except Exception:
                    parsed.append(None)

            # --- generate thinking traces for parseable questions ---
            trace_items = [(i, p) for i, p in enumerate(parsed) if p is not None]
            if trace_items:
                trace_prompts_items = [{"question": p["question"], "paths": p["paths"]}
                                       for _, p in trace_items]
                traces = llm.generate_thinking_traces_batch(trace_prompts_items)
                for (i, _), trace in zip(trace_items, traces):
                    parsed[i]["cot"] = trace

            # --- correctness filter ---
            correctness_items = [
                {
                    "combined": llm.combine_question_and_thinking_trace_with_answer(
                        p["question"], p["cot"], p["answer"]
                    ),
                    "paths": p["paths"],
                }
                for p in parsed if p is not None and "cot" in p
            ]
            correctness_results = llm.correctness_filtering_batch(correctness_items) if correctness_items else []

            # --- collect results ---
            ci = 0
            for triple, item_parsed in zip(batch, parsed):
                if item_parsed is None or "cot" not in item_parsed:
                    failed_this_pass.append(triple)
                    continue
                verdict = correctness_results[ci]
                ci += 1
                if verdict != "Yes":
                    failed_this_pass.append(triple)
                    continue

                combined = llm.combine_question_and_thinking_trace_with_answer(
                    item_parsed["question"], item_parsed["cot"], item_parsed["answer"]
                )
                results.append({
                    "id": next_id,
                    "k_hops": 1,
                    "source_concept": item_parsed["source_concept"],
                    "target_concept": item_parsed["target_concept"],
                    "paths": item_parsed["paths"],
                    "question_and_explanation": combined,
                })
                next_id += 1

            # checkpoint after every batch
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        remaining = failed_this_pass
        print(f"Pass {retry + 1} done: {len(results)} total generated, {len(remaining)} still failing")

    if remaining:
        print(f"\nWarning: {len(remaining)} triples could not be generated after {max_retries} passes:")
        for t in remaining:
            print(f"  {t}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} ungraded questions to {output_path}")
    print("Next step: run grade_curriculum_tinker.py on this file to get GPT-OSS-120b verdicts.")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dataset",        default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_hop_3.json")
    parser.add_argument("--verdicts",            default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/verdicts_gptoss120b.json")
    parser.add_argument("--kg_triples",          default="/scratch/gpfs/JHA/hp9084/curricula_gen/networks_kg/validated_triples.csv")
    parser.add_argument("--output",              default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_gap_fill.json")
    parser.add_argument("--model_name",          default="google/gemma-3-27b-it")
    parser.add_argument("--hf_cache",            default="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface")
    parser.add_argument("--tensor_parallel_size",type=int, default=2)
    parser.add_argument("--batch_size",          type=int, default=32)
    parser.add_argument("--max_retries",         type=int, default=5)
    args = parser.parse_args()

    uncovered = find_uncovered_triples(args.full_dataset, args.verdicts, args.kg_triples)
    if not uncovered:
        print("Full coverage already achieved.")
        return

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    llm = LocalLLMBackend(
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_cache_dir=args.hf_cache,
    )

    generate_for_triples(uncovered, args.output, llm, args.batch_size, args.max_retries)


if __name__ == "__main__":
    main()
