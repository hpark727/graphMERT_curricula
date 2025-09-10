'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.

Adapted from: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py
'''

import collections
from tqdm import tqdm
import datasets
import re

def load_question(question_item):

    question = re.sub(r'<Explanation>.*?</Explanation>', '', question_item['question_and_explanation'], flags=re.DOTALL)
    return question

def load_path(question_item):
    
    return question_item['paths']

def normalize_string(text):
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def word_ngrams(text, n):
    """Generate word-level n-grams from text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(documents, ngram_size=12):
    """Build ngram lookup for documents."""
    print(f"Building {ngram_size}-gram lookup...")
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)
    
    return lookup


def find_contaminated_questions(test_lookup, train_lookup):
    """Find overlapping documents based on ngram matches."""
    contaminated_ids = set()
    matched_ngrams = []  # For debugging
    
    for ngram, test_doc_ids in tqdm(test_lookup.items(), desc="Checking overlaps"):
        if ngram in train_lookup:
            contaminated_ids.update(test_doc_ids)
            matched_ngrams.append(ngram)
    
    # Print some example matches for inspection
    print('matched_ngrams', len(matched_ngrams))
    if matched_ngrams:
        print("\nExample matching n-grams:")
        for ngram in matched_ngrams[:10]: 
            print(f"  - {ngram}")
    
    return contaminated_ids

