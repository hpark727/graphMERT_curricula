"""
Non-LLM pre-filter for gap-fill MCQ candidates.

For each uncovered triple we generated up to 6 candidates. We only need 1 to
pass GPT grading, so this script scores each candidate on structural heuristics
and keeps the top-K per triple before the expensive Stage 2 run.

Scoring (all non-LLM):
  - Discard: answer letter not in {A,B,C,D}
  - Discard: any option shorter than MIN_OPTION_CHARS
  - Discard: question stem shorter than MIN_STEM_CHARS
  - Score: average pairwise unigram-overlap between options (lower = more distinct)
  - Score: mean option length (longer = more substantive)
  Combined score: 0.7 * distinctiveness + 0.3 * length_score (both normalised to [0,1])

Usage:
    python -m curriculum_generator.prefilter_gap_fill \
        --input   curriculum_generator/data/curriculum_dataset_gap_fill_redundant.json \
        --output  curriculum_generator/data/curriculum_dataset_gap_fill_prefiltered.json \
        --top_k   2
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
import argparse
from collections import defaultdict

MIN_OPTION_CHARS = 20
MIN_STEM_CHARS   = 100
VALID_ANSWERS    = {"A", "B", "C", "D"}


def extract_options(question_text):
    """Return list of 4 option strings (text only, no letter prefix)."""
    matches = re.findall(r'[A-D]\.\s+(.+?)(?=\n[A-D]\.|</Options>|$)', question_text, re.DOTALL)
    return [m.strip() for m in matches]


def extract_stem(question_text):
    """Return just the question stem (inside <Question>...</Question>)."""
    m = re.search(r'<Question>(.*?)</Question>', question_text, re.DOTALL)
    return m.group(1).strip() if m else question_text


def unigram_overlap(a, b):
    """Jaccard overlap between unigram sets of two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def distinctiveness_score(options):
    """
    Mean pairwise Jaccard similarity between all option pairs.
    Lower = more distinct options = better question.
    """
    pairs = []
    for i in range(len(options)):
        for j in range(i + 1, len(options)):
            pairs.append(unigram_overlap(options[i], options[j]))
    return sum(pairs) / len(pairs) if pairs else 1.0


def score_entry(entry):
    """Return (passes_hard_filters: bool, score: float)."""
    answer = entry.get("answer", "").strip().rstrip("\\").strip()
    if answer not in VALID_ANSWERS:
        return False, 0.0

    q = entry["question"]
    stem = extract_stem(q)
    if len(stem) < MIN_STEM_CHARS:
        return False, 0.0

    options = extract_options(q)
    if len(options) != 4:
        return False, 0.0
    if any(len(opt) < MIN_OPTION_CHARS for opt in options):
        return False, 0.0

    overlap = distinctiveness_score(options)   # 0 = perfectly distinct, 1 = identical
    distinctiveness = 1.0 - overlap            # higher = better

    mean_len = sum(len(o) for o in options) / 4
    return True, (distinctiveness, mean_len)   # tuple for later normalisation


def prefilter(input_path, output_path, top_k):
    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} candidates.")

    # Group by triple
    groups = defaultdict(list)
    malformed = 0
    for entry in data:
        passes, raw_score = score_entry(entry)
        if not passes:
            malformed += 1
            continue
        p = entry["paths"][0]
        key = (p["start"], p["relation"], p["end"])
        groups[key].append((raw_score, entry))

    print(f"Hard-filter removed: {malformed}  ({100*malformed/len(data):.1f}%)")
    print(f"Triples with ≥1 candidate: {len(groups)}")

    # Normalise scores and pick top-K per triple
    kept = []
    for key, candidates in groups.items():
        if len(candidates) == 1:
            kept.append(candidates[0][1])
            continue

        # Normalise distinctiveness and mean_len across this triple's candidates
        dists  = [s[0] for s, _ in candidates]
        lens   = [s[1] for s, _ in candidates]
        d_min, d_max = min(dists), max(dists)
        l_min, l_max = min(lens),  max(lens)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        scored = []
        for (d, l), entry in candidates:
            combined = 0.7 * norm(d, d_min, d_max) + 0.3 * norm(l, l_min, l_max)
            scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        for _, entry in scored[:top_k]:
            kept.append(entry)

    # Re-assign contiguous ids
    for i, entry in enumerate(kept):
        entry["id"] = i

    with open(output_path, "w") as f:
        json.dump(kept, f, indent=2)

    print(f"\nKept {len(kept)} candidates (top-{top_k} per triple).")
    print(f"Reduction: {len(data)} → {len(kept)}  ({100*len(kept)/len(data):.1f}% of original)")
    print(f"Saved to {output_path}")
    print(f"\nNext: python -m curriculum_generator.generate_thinking_traces_tinker "
          f"--input {output_path} "
          f"--output curriculum_generator/data/curriculum_dataset_gap_fill_final.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="curriculum_generator/data/curriculum_dataset_gap_fill_redundant.json")
    parser.add_argument("--output", default="curriculum_generator/data/curriculum_dataset_gap_fill_prefiltered.json")
    parser.add_argument("--top_k",  type=int, default=2)
    args = parser.parse_args()
    prefilter(args.input, args.output, args.top_k)


if __name__ == "__main__":
    main()
