"""
Local version of generate_test_set using GPT-OSS-120b via Tinker.
PathGenerator runs locally (loads KG from disk); questions generated concurrently via API.
No thinking traces — run generate_thinking_traces_tinker.py on the output.

Usage:
    python -m curriculum_generator.generate_test_set_tinker \
        --output        curriculum_generator/data/test_set_raw.json \
        --num_per_hop   3000 \
        --hop_counts    3 4 5 \
        --concurrency   16
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from curriculum_generator.generate_questions_tinker import TinkerLLMBackend, _shuffle_answer_options
from curriculum_generator.generate_questions_local import PathGenerator


def sample_paths(path_generator, hop_counts, num_per_hop):
    """Sample paths for all hop counts, returning a shuffled list of (k, path_dict)."""
    items = []
    for k in hop_counts:
        needed = num_per_hop
        attempts = 0
        while len([i for i in items if i[0] == k]) < needed and attempts < needed * 3:
            try:
                path = path_generator.generate_paths(k_hops=k)
                items.append((k, path))
            except ValueError:
                pass
            attempts += 1
    random.shuffle(items)
    return items


def generate_one(k, path, backend):
    raw = backend.generate_question(
        source_concept=path["source_concept"],
        target_concept=path["target_concept"],
        paths=path["paths"],
    )
    if not raw or "<Answer>" not in raw:
        return None
    question_text, answer = backend.separate_question_and_answer(raw)
    question_text, answer = _shuffle_answer_options(question_text, answer)
    if not backend.quality_filtering(question_text):
        return None
    return {
        "k_hops": k,
        "source_concept": path["source_concept"],
        "target_concept": path["target_concept"],
        "paths": path["paths"],
        "question": question_text,
        "answer": answer,
    }


def run(output_path, num_per_hop, hop_counts, concurrency):
    results = []
    next_id = 0
    hop_done = {k: 0 for k in hop_counts}

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        next_id = max((r["id"] for r in results), default=-1) + 1
        for r in results:
            if r["k_hops"] in hop_done:
                hop_done[r["k_hops"]] += 1
        print(f"Resuming: {len(results)} already generated. Per-hop: {hop_done}")

    base_dir = os.path.join(os.path.dirname(__file__), "data_kg", "networks_kg")
    path_generator = PathGenerator(
        vocab_path=os.path.join(base_dir, "vocab.txt"),
        graph_path=os.path.join(base_dir, "networks.graph"),
        categories_path=os.path.join(base_dir, "categories.json"),
    )

    # Sample paths for remaining quota
    remaining_hops = [k for k in hop_counts if hop_done[k] < num_per_hop]
    remaining_per_hop = {k: num_per_hop - hop_done[k] for k in remaining_hops}

    todo = []
    for k in remaining_hops:
        needed = remaining_per_hop[k]
        sampled = 0
        attempts = 0
        while sampled < needed and attempts < needed * 3:
            try:
                path = path_generator.generate_paths(k_hops=k)
                todo.append((k, path))
                sampled += 1
            except ValueError:
                pass
            attempts += 1

    random.shuffle(todo)
    print(f"Sampled {len(todo)} paths. Generating MCQs with {concurrency} workers...")

    _local = threading.local()

    def get_backend():
        if not hasattr(_local, "llm"):
            _local.llm = TinkerLLMBackend(model_name="openai/gpt-oss-120b")
        return _local.llm

    lock = threading.Lock()
    checkpoint_counter = [0]
    failed = [0]

    def worker(k_and_path):
        k, path = k_and_path
        try:
            result = generate_one(k, path, get_backend())
        except Exception as e:
            tqdm.write(f"[error] {k}-hop {path['source_concept']}: {e}")
            result = None

        with lock:
            if result is not None:
                results.append({"id": next_id + len(results), **result})
            else:
                failed[0] += 1
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % 100 == 0:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                hop_summary = {k: sum(1 for r in results if r["k_hops"] == k) for k in hop_counts}
                tqdm.write(f"[checkpoint] {len(results)} generated | {hop_summary}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(worker, item): item for item in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Test set MCQ"):
            future.result()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    hop_summary = {k: sum(1 for r in results if r["k_hops"] == k) for k in hop_counts}
    print(f"\nSaved {len(results)} MCQ entries to {output_path}")
    print(f"Per-hop: {hop_summary}")
    print("Next: run generate_thinking_traces_tinker.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",       default="curriculum_generator/data/test_set_raw.json")
    parser.add_argument("--num_per_hop",  type=int, default=3000)
    parser.add_argument("--hop_counts",   type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--concurrency",  type=int, default=16)
    args = parser.parse_args()

    run(args.output, args.num_per_hop, tuple(args.hop_counts), args.concurrency)


if __name__ == "__main__":
    main()
