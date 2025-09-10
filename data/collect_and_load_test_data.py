'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import datasets
import os
import json


def load_icd_bench_questions():
    try:
        dataset = datasets.load_dataset(
            "bottom-up-superintelligence/ICD-Bench", 
            trust_remote_code=True,
            split='test'
        )
        return [f"{row['question']}\n{row['options']}" for row in dataset]

    except Exception as e:
        print(f"Error loading ICD-Bench dataset: {e}")
        
def load_icd_bench_paths():
    try:
        dataset = datasets.load_dataset(
            "bottom-up-superintelligence/ICD-Bench", 
            trust_remote_code=True,
            split='test'
        )
        return [row['path'] for row in dataset]
    except Exception as e:
        print(f"Error loading ICD-Bench paths: {e}")
