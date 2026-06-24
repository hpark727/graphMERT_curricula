"""
Local version of fill_coverage_gaps_redundant using GPT-OSS-120b via Tinker.
Generates num_per_triple MCQ candidates per uncovered triple concurrently.
No thinking traces — run generate_thinking_traces_tinker.py on the output.

Usage:
    python -m curriculum_generator.fill_coverage_gaps_redundant_tinker \
        --full_dataset  curriculum_generator/data/curriculum_dataset_hop_3.json \
        --verdicts      curriculum_generator/data/verdicts_gptoss120b.json \
        --kg_triples    networks_kg/validated_triples.csv \
        --output        curriculum_generator/data/curriculum_dataset_gap_fill_redundant.json \
        --num_per_triple 10 \
        --concurrency   16
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import csv
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from curriculum_generator.generate_questions_tinker import TinkerLLMBackend, _shuffle_answer_options


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


def generate_one(item, backend):
    head = item["source_concept"]
    tail = item["target_concept"]
    paths = item["paths"]

    raw = backend.generate_question(source_concept=head, target_concept=tail, paths=paths)
    if not raw or "<Answer>" not in raw:
        return None
    question_text, answer = backend.separate_question_and_answer(raw)
    question_text, answer = _shuffle_answer_options(question_text, answer)
    if not backend.quality_filtering(question_text):
        return None
    return {
        "source_concept": head,
        "target_concept": tail,
        "paths": paths,
        "question": question_text,
        "answer": answer,
    }


def run(uncovered, output_path, num_per_triple, concurrency):
    results = []
    next_id = 0
    counts = {}

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        next_id = max((r["id"] for r in results), default=-1) + 1
        for r in results:
            p = r["paths"][0]
            key = (p["start"], p["relation"], p["end"])
            counts[key] = counts.get(key, 0) + 1
        uncovered = [t for t in uncovered if counts.get(t, 0) < num_per_triple]
        print(f"Resuming: {len(results)} already generated. "
              f"{len(uncovered)} triples still need more candidates.")

    todo = []
    for triple in uncovered:
        head, rel, tail = triple
        already = counts.get(triple, 0)
        for _ in range(num_per_triple - already):
            todo.append({
                "source_concept": head,
                "target_concept": tail,
                "paths": [{"start": head, "relation": rel, "end": tail}],
            })

    print(f"Total generation calls: {len(todo)} ({len(uncovered)} triples × up to {num_per_triple})")

    _local = threading.local()

    def get_backend():
        if not hasattr(_local, "llm"):
            _local.llm = TinkerLLMBackend(model_name="openai/gpt-oss-120b")
        return _local.llm

    lock = threading.Lock()
    checkpoint_counter = [0]
    failed = [0]

    def worker(item):
        try:
            result = generate_one(item, get_backend())
        except Exception as e:
            tqdm.write(f"[error] {item['source_concept']} -> {item['target_concept']}: {e}")
            result = None

        with lock:
            if result is not None:
                results.append({
                    "id": next_id + len(results),
                    "k_hops": 1,
                    **result,
                })
            else:
                failed[0] += 1
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % 100 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                tqdm.write(f"[checkpoint] {len(results)} generated, {failed[0]} failed")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(worker, item): item for item in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Gap fill MCQ"):
            future.result()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} MCQ candidates to {output_path}")
    print("Next: run generate_thinking_traces_tinker.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_dataset",   default="curriculum_generator/data/curriculum_dataset_hop_3.json")
    parser.add_argument("--verdicts",       default="curriculum_generator/data/verdicts_gptoss120b.json")
    parser.add_argument("--kg_triples",     default="networks_kg/validated_triples.csv")
    parser.add_argument("--output",         default="curriculum_generator/data/curriculum_dataset_gap_fill_redundant.json")
    parser.add_argument("--num_per_triple", type=int, default=10)
    parser.add_argument("--concurrency",    type=int, default=16)
    args = parser.parse_args()

    uncovered = find_uncovered_triples(args.full_dataset, args.verdicts, args.kg_triples)
    if not uncovered:
        print("Full coverage already achieved.")
        return
    run(uncovered, args.output, args.num_per_triple, args.concurrency)


if __name__ == "__main__":
    main()
