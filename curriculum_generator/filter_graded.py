'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import json
import argparse
from collections import Counter


def filter_graded(dataset_path: str, verdict_paths: list, output_path: str) -> None:
    with open(dataset_path) as f:
        dataset = json.load(f)
    print(f"Input: {len(dataset)} questions")

    all_verdicts = []
    for path in verdict_paths:
        with open(path) as f:
            v = json.load(f)
        all_verdicts.append(v)
        yes = sum(1 for x in v.values() if x == "Yes")
        print(f"  {path}: {yes}/{len(v)} Yes ({100*yes/len(v):.1f}%)")

    kept = []
    for entry in dataset:
        eid = str(entry['id'])
        if all(v.get(eid) == "Yes" for v in all_verdicts):
            kept.append(entry)

    hop_counts = Counter(e['k_hops'] for e in kept)
    print(f"\nKept: {len(kept)}/{len(dataset)} ({100*len(kept)/len(dataset):.1f}%)")
    for k in sorted(hop_counts):
        print(f"  {k}-hop: {hop_counts[k]}")

    with open(output_path, 'w') as f:
        json.dump(kept, f, indent=2)
    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter curriculum dataset to questions passing all graders."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to curriculum_dataset_hop_1_2.json")
    parser.add_argument("--verdicts", nargs='+', required=True,
                        help="Paths to verdicts_<grader>.json files (one per grader)")
    parser.add_argument("--output", required=True,
                        help="Output path, e.g. curriculum_dataset_verified.json")
    args = parser.parse_args()

    filter_graded(args.dataset, args.verdicts, args.output)


if __name__ == "__main__":
    main()
