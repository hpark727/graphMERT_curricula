'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from curriculum_generator.generate_questions_local import QAGeneratorLocal
import json
import random
import argparse
from tqdm import tqdm


def generate_curriculum_local(
    max_k_hops: int,
    num_questions: int,
    output_dir: str,
    model_name: str,
    tensor_parallel_size: int = 1,
    domain: str = 'computer networking',
    hf_cache_dir: str = None,
    batch_size: int = 8,
    error_threshold: int = 10,
) -> None:

    output_file = os.path.join(output_dir, f"curriculum_dataset_hop_{max_k_hops}.json")
    dataset = []
    question_idx = 0

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            dataset = json.load(f)
            if dataset:
                question_idx = max(entry['id'] for entry in dataset) + 1
                print(f"Resuming from question {question_idx}")

    qa_gym = QAGeneratorLocal(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        domain=domain,
        hf_cache_dir=hf_cache_dir,
    )

    error_window = 0
    errors_total = 0

    pbar = tqdm(total=num_questions, initial=question_idx,
                desc="Generating", unit="q", dynamic_ncols=True)

    while question_idx < num_questions:

        if error_window > error_threshold:
            pbar.close()
            raise RuntimeError("Too many consecutive errors. Stopping.")

        # --- batch path: sample a batch of candidates, filter in one pass ---
        k_hops = random.randint(1, max_k_hops)
        try:
            candidates = qa_gym.generate_questions_batch(
                k_hops=k_hops, batch_size=batch_size
            )
        except Exception as e:
            error_window += 1
            errors_total += 1
            pbar.write(f"[skip] batch path generation failed: {e}")
            continue

        for question_data in candidates:
            if question_idx >= num_questions:
                break

            try:
                quality = qa_gym.quality_filtering(question_data['question'])
                if not quality:
                    raise ValueError("Quality filtering failed.")

                explanation = qa_gym.generate_thinking_trace(
                    question=question_data['question'],
                    paths=question_data['paths'],
                )
                combined = qa_gym.combine_question_and_thinking_trace_with_answer(
                    question=question_data['question'],
                    explanation=explanation,
                    answer=question_data['answer'],
                )
                correctness = qa_gym.correctness_filtering(combined, question_data['paths'])
                if correctness != "Yes":
                    raise ValueError(f"Correctness filtering failed ({correctness}).")

                dataset.append({
                    "id": question_idx,
                    "k_hops": k_hops,
                    "source_concept": question_data['source_concept'],
                    "target_concept": question_data['target_concept'],
                    "paths": question_data['paths'],
                    "question_and_explanation": combined,
                })

                for path in question_data['paths']:
                    qa_gym.generator.vocab_freq[path['start']] += 1

                question_idx += 1
                error_window = 0

                pbar.update(1)
                pbar.set_postfix(errors=errors_total, hops=k_hops,
                                 src=question_data['source_concept'][:20])

                if question_idx % 50 == 0:
                    with open(output_file, 'w') as f:
                        json.dump(dataset, f, indent=2)
                    with open(os.path.join(output_dir, "nodes_freq.json"), 'w') as f:
                        json.dump(qa_gym.generator.vocab_freq, f, indent=2)
                    pbar.write(f"[checkpoint] {question_idx} questions saved.")

            except Exception as e:
                error_window += 1
                errors_total += 1
                pbar.set_postfix(errors=errors_total)
                pbar.write(f"[skip] {e}")

    pbar.close()

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Done. {len(dataset)} questions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_k_hops", type=int, default=3)
    parser.add_argument("--num_questions", type=int, default=24000)
    parser.add_argument("--output_dir", type=str, default="/scratch/gpfs/$USER/curriculum_training_data/")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Number of GPUs to shard the model across (vLLM tensor parallelism)")
    parser.add_argument("--domain", type=str, default="computer networking")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="Override HuggingFace cache dir (e.g. /scratch/gpfs/$USER/.cache/huggingface)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of questions to generate per vLLM batch call")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generate_curriculum_local(
        max_k_hops=args.max_k_hops,
        num_questions=args.num_questions,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        domain=args.domain,
        hf_cache_dir=args.hf_cache_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
