"""
Generate multiple candidate Q&A pairs per uncovered KG triple so that GPT-OSS
validation has enough candidates to pass at least one per triple.

Unlike fill_coverage_gaps.py this script:
  - Generates num_per_triple questions per uncovered triple (default 10)
  - Skips Gemma's own correctness filter (GPT-OSS grades everything afterward)
  - Saves ALL structurally-valid questions, not just the first passing one

Output is ungraded. Run grade_curriculum_tinker.py on it afterward.

Usage on della:
    python -m curriculum_generator.fill_coverage_gaps_redundant \
        --full_dataset  /scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_hop_3.json \
        --verdicts      /scratch/gpfs/JHA/hp9084/curricula_gen/curriculum_generator/data/verdicts_gptoss120b.json \
        --kg_triples    /scratch/gpfs/JHA/hp9084/curricula_gen/networks_kg/validated_triples.csv \
        --output        /scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_gap_fill_redundant.json \
        --model_name    google/gemma-3-27b-it \
        --hf_cache      /scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface \
        --tensor_parallel_size 2 \
        --batch_size    64 \
        --num_per_triple 10
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import argparse
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
            k = int(row.get("k_hops", 1)) if "k_hops" in row else None
            all_triples.append((
                row["head"].strip(),
                row["relation"].strip(),
                row["tail"].strip(),
            ))

    uncovered = [t for t in all_triples if t not in covered]

    print(f"KG triples total   : {len(all_triples)}")
    print(f"Covered by GPT-yes : {len(covered & set(all_triples))} "
          f"({100 * len(covered & set(all_triples)) / len(all_triples):.1f}%)")
    print(f"Uncovered          : {len(uncovered)}")
    return uncovered


def generate_redundant(uncovered, output_path, llm, batch_size, num_per_triple):
    results = []
    next_id = 0

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        next_id = max((r["id"] for r in results), default=-1) + 1

        # Count how many we already have per triple
        counts = {}
        for r in results:
            p = r["paths"][0]
            key = (p["start"], p["relation"], p["end"])
            counts[key] = counts.get(key, 0) + 1

        # Only keep triples that still need more candidates
        uncovered = [t for t in uncovered if counts.get(t, 0) < num_per_triple]
        print(f"Resuming: {len(results)} already generated. "
              f"{len(uncovered)} triples still need more candidates.")
    else:
        counts = {}

    # Expand: each triple appears num_per_triple times (minus what we have)
    todo_items = []
    for triple in uncovered:
        already = counts.get(triple, 0)
        remaining = num_per_triple - already
        head, rel, tail = triple
        for _ in range(remaining):
            todo_items.append({
                "source_concept": head,
                "target_concept": tail,
                "paths": [{"start": head, "relation": rel, "end": tail}],
            })

    print(f"Total generation calls: {len(todo_items)} "
          f"({len(uncovered)} triples × up to {num_per_triple} each)")

    for batch_start in tqdm(range(0, len(todo_items), batch_size), desc="Generating"):
        batch = todo_items[batch_start: batch_start + batch_size]

        # --- generate questions ---
        raw_questions = llm.generate_questions_batch(batch)

        # --- parse + structural quality filter ---
        parsed = []
        for item, raw in zip(batch, raw_questions):
            try:
                question_text, answer = llm.separate_question_and_answer(raw)
                question_text, answer = _shuffle_answer_options(question_text, answer)
                if not llm.quality_filtering(question_text):
                    parsed.append(None)
                    continue
                parsed.append({**item, "question": question_text, "answer": answer})
            except Exception:
                parsed.append(None)

        # --- collect results (thinking traces generated separately by GPT-OSS) ---
        for item_parsed in parsed:
            if item_parsed is None:
                continue
            results.append({
                "id": next_id,
                "k_hops": 1,
                "source_concept": item_parsed["source_concept"],
                "target_concept": item_parsed["target_concept"],
                "paths": item_parsed["paths"],
                "question": item_parsed["question"],
                "answer": item_parsed["answer"],
            })
            next_id += 1

        if (batch_start // batch_size) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} question-only candidates to {output_path}")
    print("Next: run generate_thinking_traces_tinker.py to add GPT-OSS thinking traces and grade.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dataset",         default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_hop_3.json")
    parser.add_argument("--verdicts",             default="/scratch/gpfs/JHA/hp9084/curricula_gen/curriculum_generator/data/verdicts_gptoss120b.json")
    parser.add_argument("--kg_triples",           default="/scratch/gpfs/JHA/hp9084/curricula_gen/networks_kg/validated_triples.csv")
    parser.add_argument("--output",               default="/scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_gap_fill_redundant.json")
    parser.add_argument("--model_name",           default="google/gemma-3-27b-it")
    parser.add_argument("--hf_cache",             default="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--batch_size",           type=int, default=64)
    parser.add_argument("--num_per_triple",       type=int, default=10)
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

    generate_redundant(uncovered, args.output, llm, args.batch_size, args.num_per_triple)


if __name__ == "__main__":
    main()
