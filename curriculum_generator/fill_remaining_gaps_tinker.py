"""
Identify triples uncovered by curriculum_dataset_final.json and generate
fresh MCQ candidates for them via GPT-OSS-120b through Tinker.

No thinking traces — run generate_thinking_traces_tinker.py on the output.

Usage:
    python -m curriculum_generator.fill_remaining_gaps_tinker \
        --final_dataset curriculum_generator/data/curriculum_dataset_final.json \
        --kg_triples    networks_kg/validated_triples.csv \
        --output        curriculum_generator/data/remaining_gaps_raw.json \
        --num_per_triple 6 \
        --concurrency   16
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import csv, json, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from curriculum_generator.generate_questions_tinker import TinkerLLMBackend, _shuffle_answer_options


def find_uncovered(final_dataset_path, kg_triples_path):
    with open(final_dataset_path) as f:
        data = json.load(f)

    covered = set()
    for e in data:
        for hop in e['paths']:
            covered.add((hop['start'], hop['relation'], hop['end']))

    all_triples = []
    with open(kg_triples_path) as f:
        for row in csv.DictReader(f):
            all_triples.append((row['head'].strip(), row['relation'].strip(), row['tail'].strip()))

    uncovered = [t for t in all_triples if t not in covered]
    print(f"KG triples total  : {len(all_triples)}")
    print(f"Covered           : {len(covered & set(all_triples))}")
    print(f"Uncovered         : {len(uncovered)}")
    return uncovered


def generate_one(item, backend):
    head, rel, tail = item['source_concept'], item['relation'], item['target_concept']
    paths = [{'start': head, 'relation': rel, 'end': tail}]

    raw = backend.generate_question(source_concept=head, target_concept=tail, paths=paths)
    if not raw or '<Answer>' not in raw:
        return None
    question_text, answer = backend.separate_question_and_answer(raw)
    question_text, answer = _shuffle_answer_options(question_text, answer)
    if not backend.quality_filtering(question_text):
        return None
    return {
        'source_concept': head,
        'target_concept': tail,
        'relation':       rel,
        'paths':          paths,
        'question':       question_text,
        'answer':         answer,
    }


def run(uncovered, output_path, num_per_triple, concurrency):
    results = []
    counts = {}
    next_id = 0

    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        next_id = max((r['id'] for r in results), default=-1) + 1
        for r in results:
            p = r['paths'][0]
            key = (p['start'], p['relation'], p['end'])
            counts[key] = counts.get(key, 0) + 1
        uncovered = [t for t in uncovered if counts.get(t, 0) < num_per_triple]
        print(f"Resuming: {len(results)} generated, {len(uncovered)} triples still need candidates.")

    todo = []
    for triple in uncovered:
        head, rel, tail = triple
        already = counts.get(triple, 0)
        for _ in range(num_per_triple - already):
            todo.append({'source_concept': head, 'relation': rel, 'target_concept': tail})

    print(f"Total generation calls: {len(todo)}")

    _local = threading.local()

    def get_backend():
        if not hasattr(_local, 'llm'):
            _local.llm = TinkerLLMBackend(model_name='openai/gpt-oss-120b')
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
                results.append({'id': next_id + len(results), 'k_hops': 1, **result})
            else:
                failed[0] += 1
            checkpoint_counter[0] += 1
            if checkpoint_counter[0] % 50 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                tqdm.write(f"[checkpoint] {len(results)} generated, {failed[0]} failed")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(worker, item): item for item in todo}
        for future in tqdm(as_completed(futures), total=len(todo), desc="Remaining gaps"):
            future.result()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} candidates to {output_path}")
    print("Next: run prefilter_gap_fill.py then generate_thinking_traces_tinker.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final_dataset',  default='curriculum_generator/data/curriculum_dataset_final.json')
    parser.add_argument('--kg_triples',     default='networks_kg/validated_triples.csv')
    parser.add_argument('--output',         default='curriculum_generator/data/remaining_gaps_raw.json')
    parser.add_argument('--num_per_triple', type=int, default=6)
    parser.add_argument('--concurrency',    type=int, default=16)
    args = parser.parse_args()

    uncovered = find_uncovered(args.final_dataset, args.kg_triples)
    if not uncovered:
        print("Full coverage already achieved.")
        return
    run(uncovered, args.output, args.num_per_triple, args.concurrency)


if __name__ == '__main__':
    main()
