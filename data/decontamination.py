'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import json
from collect_and_load_test_data import *
from decontamination_utils import *
from argparse import ArgumentParser

def main(args):

    with open(args.train_questions_path, "r") as f:
        train_data = json.load(f)

    train_questions = [load_question(question) for question in train_data]
    train_paths = [load_path(question) for question in train_data]
    test_questions = []
    test_paths = []
    # Get all data loading functions from collect_and_load_test_data
    question_loading_functions = [
        load_icd_bench_questions,
    ]
    
    path_loading_functions = [
        load_icd_bench_paths,
    ]
    
    print("Performing n-gram decontamination for questions...")
    for load_fn in question_loading_functions:
        try:
            questions = load_fn()
            if questions:
                test_questions.extend(questions)
        except Exception as e:
            print(f"Error loading from {load_fn.__name__}: {e}")

    test_lookup = build_ngram_lookup(test_questions, ngram_size=args.ngram_size)
    train_lookup = build_ngram_lookup(train_questions, ngram_size=args.ngram_size)
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)


    print("Original number of train questions:", len(train_data))
    print("Number of contaminated questions:", len(contaminated_ids))
    
    # Save contaminated n-gram overlaps
    contaminated_overlaps = []
    for ngram, test_doc_ids in test_lookup.items():
        if ngram in train_lookup:
            contaminated_overlaps.append({
                'ngram': ngram,
                'test_doc_ids': list(test_doc_ids),
                'train_doc_ids': list(train_lookup[ngram])
            })
    
    overlaps_path = args.train_questions_path.replace(".json", "_contaminated_overlaps.json")
    with open(overlaps_path, "w") as f:
        json.dump(contaminated_overlaps, f, indent=2)
    
    print(f"Saved contaminated n-gram overlaps to {overlaps_path}")
    
    print("Performing path decontamination for questions...")
    
    for load_fn in path_loading_functions:
        try:
            test_paths.extend(load_fn())
        except Exception as e:
            print(f"Error loading from test paths: {e}")
            
     
    contaminated_path_ids = set()
    for i, path in enumerate(train_paths):
        if path in test_paths:
            contaminated_path_ids.add(i)
            
            
    print("Number of contaminated paths:", len(contaminated_path_ids))
    
    contaminated_ids = contaminated_ids.union(contaminated_path_ids)
    not_contaminated_questions = set(range(len(train_data))) - contaminated_ids
    
    train_data_decontaminated = [train_data[i] for i in not_contaminated_questions]
    
    with open(args.train_questions_path.replace(".json", "_decontaminated.json"), "w") as f:
        json.dump(train_data_decontaminated, f)

    print(f"Saved decontaminated train questions to {args.train_questions_path.replace('.json', '_decontaminated.json')}")



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--train_questions_path", type=str, default="/curriculum_training_data/curriculum_dataset_hop_3.json")
    parser.add_argument("--ngram_size", type=int, default=18)
    args = parser.parse_args()
    main(args)
