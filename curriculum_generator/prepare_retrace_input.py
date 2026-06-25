"""
Extract question + answer from existing question_and_explanation entries
and write a flat JSON suitable for generate_thinking_traces_tinker.py.

Usage:
    python -m curriculum_generator.prepare_retrace_input \
        --input  curriculum_generator/data/curriculum_dataset_hop_1_2_decontaminated.json \
        --output curriculum_generator/data/hop_1_2_questions_only.json
"""

import os, sys, re, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_question_and_answer(qae: str):
    """Split question_and_explanation back into (question, answer)."""
    # question = everything before <Explanation>
    exp_start = qae.find('<Explanation>')
    if exp_start == -1:
        return None, None
    question = qae[:exp_start].strip()

    # answer = content between <Answer>: and </Answer> (the final answer block)
    m = re.search(r'<Answer>:\s*([A-D])\s*</Answer>', qae)
    if not m:
        # fallback: last <Answer> tag without colon
        m = re.search(r'<Answer>\s*([A-D])\s*</Answer>', qae)
    if not m:
        return None, None
    answer = m.group(1).strip()
    return question, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    required=True)
    parser.add_argument('--output',   required=True)
    parser.add_argument('--verdicts', default=None,
                        help='Optional verdicts JSON; if provided, only keep entries with verdict==Yes')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    passed_ids = None
    if args.verdicts:
        with open(args.verdicts) as f:
            verdicts = json.load(f)
        passed_ids = {int(k) for k, v in verdicts.items() if v == 'Yes'}
        print(f'Verdicts loaded: {len(passed_ids)} passing ids')

    out, skipped = [], 0
    for e in data:
        if passed_ids is not None and e['id'] not in passed_ids:
            skipped += 1
            continue
        question, answer = parse_question_and_answer(e['question_and_explanation'])
        if not question or not answer:
            skipped += 1
            continue
        out.append({
            'id':             e['id'],
            'k_hops':         e['k_hops'],
            'source_concept': e['source_concept'],
            'target_concept': e['target_concept'],
            'paths':          e['paths'],
            'question':       question,
            'answer':         answer,
        })

    print(f'Parsed {len(out)} entries, skipped {skipped}')
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
