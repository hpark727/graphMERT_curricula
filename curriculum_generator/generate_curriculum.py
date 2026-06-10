'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))
from curriculum_generator.generate_questions_tinker import QAGenerator
import json
from typing import List
import time
from tqdm import tqdm
from datetime import datetime
from collections import deque
import argparse
import random
    

def generate_curriculum(
    max_k_hops: int,
    num_questions: int,
    output_dir: str,
    api_key: str,
    backend: str = 'tinker',
    model_name: str = 'openai/gpt-oss-120b',
    domain: str = 'computer networking',
    error_threshold: int = 10,
    sleep_threshold: int = 0,
    ) -> None:
    """
    Generate a dataset of curriculum questions from a knowledge graph and save them locally.
    
    Args:
        max_k_hops: Maximum hop length for path generation
        num_questions: Number of questions to generate
        api_key: API key for the LLM
        output_dir: Directory to save the generated questions
        error_threshold: Number of errors to tolerate before restarting
        sleep_threshold: Number of seconds to sleep before restarting to prevent being rate limitied by API
    """
    
    output_file = os.path.join(output_dir, f"curriculum_dataset_hop_{max_k_hops}.json")
    dataset = []
    question_idx = 0
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            dataset = json.load(f)
            if dataset:
                question_idx = max(entry['id'] for entry in dataset) + 1
                print(f"Resuming from question {question_idx}+1")
    
    qa_gym = QAGenerator(api_key=api_key, model_name=model_name, domain=domain)
    error_window = 0
    errors_total = 0

    pbar = tqdm(
        total=num_questions,
        initial=question_idx,
        desc="Generating",
        unit="q",
        dynamic_ncols=True,
    )

    while question_idx < num_questions:

        if error_window > error_threshold:
            pbar.close()
            raise Exception("Too many consecutive errors. Stopping script.")

        try:
            k_hops = random.randint(1, max_k_hops)

            question_data = qa_gym.generate_questions(k_hops=k_hops)

            quality = qa_gym.quality_filtering(question_data['question'])
            if not quality:
                raise Exception("Quality filtering failed.")

            explanation = qa_gym.llm.generate_thinking_trace(
                question=question_data['question'],
                paths=question_data['paths']
            )

            combined = qa_gym.llm.combine_question_and_thinking_trace_with_answer(
                question=question_data['question'],
                explanation=explanation,
                answer=question_data['answer']
            )
            correctness = qa_gym.correctness_filtering(combined, question_data['paths'])
            if not correctness:
                raise Exception("Correctness filtering failed.")

            question_entry = {
                "id": question_idx,
                "k_hops": k_hops,
                "source_concept": question_data['source_concept'],
                "target_concept": question_data['target_concept'],
                "paths": question_data['paths'],
                "question_and_explanation": combined
            }

            dataset.append(question_entry)
            question_idx += 1
            error_window = 0

            for path in question_data['paths']:
                qa_gym.generator.vocab_freq[path['start']] += 1

            pbar.update(1)
            pbar.set_postfix(errors=errors_total, hops=k_hops,
                             src=question_data['source_concept'][:20])

            if question_idx % 5 == 0:
                with open(output_file, 'w') as f:
                    json.dump(dataset, f, indent=2)
                with open(os.path.join(output_dir, "nodes_freq.json"), 'w') as f:
                    json.dump(qa_gym.generator.vocab_freq, f, indent=2)
                pbar.write(f"[checkpoint] {question_idx} questions saved to {output_file}")

            time.sleep(sleep_threshold)

        except Exception as e:
            error_window += 1
            errors_total += 1
            pbar.set_postfix(errors=errors_total)
            pbar.write(f"[skip] {e}")
            continue

    pbar.close()

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Done. {len(dataset)} questions saved to {output_file}")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_k_hops", type=int, default=3)
    parser.add_argument("--num_questions", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="/curriculum_training_data/")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--backend", type=str, default="tinker", choices=["tinker", "gemini"])
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--domain", type=str, default="computer networking")
    args = parser.parse_args()

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    generate_curriculum(
        args.max_k_hops, args.num_questions, args.output_dir, args.api_key,
        backend=args.backend, model_name=args.model_name, domain=args.domain
    )

    

if __name__ == "__main__":
    main() 