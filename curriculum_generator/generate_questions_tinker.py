'''
Copyright (c) 2024 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

import networkx as nx
import json
import random
import pickle
import asyncio
import os
import re
from typing import List, Dict, Tuple, Optional
import tinker
from tinker import types as tinker_types


class PathGenerator:
    def __init__(self, vocab_path: str, graph_path: str, categories_path: str, vocab_freq_path: str = None):
        self.vocab = self._load_vocab(vocab_path)
        self.concept2id = {w: i for i, w in enumerate(self.vocab)}
        self.graph = self._load_graph(graph_path)
        self.categories = self._load_categories(categories_path)
        self.vocab_freq = {v: 0 for v in self.vocab}
        if vocab_freq_path is not None and os.path.exists(vocab_freq_path):
            self._update_vocab_freq(vocab_freq_path)
            print("Updated vocab freq")
        self.relations = [
            'causes',
            'communicates_with',
            'connects_to',
            'contains',
            'enables',
            'forwards',
            'has_part',
            'identifies',
            'implements',
            'increases',
            'is_a',
            'operates_at',
            'part_of',
            'provides',
            'receives',
            'requires',
            'runs_on',
            'sends',
            'supports',
        ]
        self.hierarchy_relations = set()

        degrees = sorted(dict(self.graph.degree()).values())
        hub_threshold = degrees[int(0.99 * len(degrees))]
        self.hub_nodes = {n for n, d in dict(self.graph.degree()).items() if d >= hub_threshold}

        self.weak_relations = {'is_a', 'part_of', 'has_part', 'contains', 'supports'}
        self.weak_relation_weight = 0.1

    def _update_vocab_freq(self, path: str):
        with open(path, 'r') as f:
            saved = json.load(f)
        for k, v in saved.items():
            if k in self.vocab_freq:
                self.vocab_freq[k] = v

    def _load_vocab(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            return f.read().splitlines()

    def _load_graph(self, path: str) -> nx.Graph:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_categories(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _get_k_hop_neighbors(self, start_node: int, k: int) -> Tuple[List[Tuple[int, int, str]], bool]:
        paths = []
        visited = {start_node}
        current = start_node

        for _ in range(k):
            neighbors = list(self.graph.neighbors(current))
            available = [n for n in neighbors if n not in visited and n not in self.hub_nodes]
            if not available:
                # Fallback: allow hub nodes rather than dead-ending the path
                available = [n for n in neighbors if n not in visited]
            if not available:
                return [], False
            weights = [
                self.weak_relation_weight
                if self.relations[self.graph[current][n][0]['rel'] % len(self.relations)]
                in self.weak_relations else 1.0
                for n in available
            ]
            neighbor = random.choices(available, weights=weights, k=1)[0]
            visited.add(neighbor)
            rel_idx = self.graph[current][neighbor][0]['rel'] % len(self.relations)
            paths.append((current, neighbor, self.relations[rel_idx]))
            current = neighbor

        return paths, True

    def generate_paths(self, category: str = None, k_hops: int = 1) -> Dict:
        if category is not None:
            concepts_in_category = self.categories[category]

        max_attempts = 10
        for _ in range(max_attempts):
            if category is not None:
                sampled = random.sample(concepts_in_category, 1)[0]
            else:
                concepts = list(self.vocab_freq.keys())
                inv = [1.0 / (f + 1e-10) for f in self.vocab_freq.values()]
                total = sum(inv)
                probs = [v / total for v in inv]
                sampled = random.choices(concepts, weights=probs, k=1)[0]

            if sampled not in self.concept2id:
                continue

            paths, ok = self._get_k_hop_neighbors(self.concept2id[sampled], k_hops)
            if ok:
                return {
                    'source_concept': sampled,
                    'target_concept': self.vocab[paths[-1][1]],
                    'paths': [
                        {'start': self.vocab[s], 'end': self.vocab[e], 'relation': r}
                        for s, e, r in paths
                    ],
                }

        raise ValueError(f"Could not find valid path after {max_attempts} attempts")


class TinkerLLMBackend:
    def __init__(self, model_name: str = 'openai/gpt-oss-120b', domain: str = 'computer networking'):
        self.domain = domain
        self.service_client = tinker.ServiceClient()  # reads TINKER_API_KEY from env
        self.sampling_client = self.service_client.create_sampling_client(base_model=model_name)
        self.tokenizer = self.sampling_client.get_tokenizer()

    def _generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Optional[str]:
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_input = tinker_types.ModelInput.from_ints(self.tokenizer.encode(chat_text))
        params = tinker_types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

        async def _run():
            result = await self.sampling_client.sample_async(
                prompt=prompt_input, sampling_params=params, num_samples=1
            )
            return self.tokenizer.decode(result.sequences[0].tokens)

        raw = asyncio.run(_run())
        # gpt-oss-120b emits a scratchpad block before the final answer;
        # extract only the last <|channel|>final<|message|> segment.
        marker = '<|channel|>final<|message|>'
        if marker in raw:
            raw = raw.split(marker)[-1]
        # strip any trailing end-of-turn tokens
        raw = raw.split('<|end|>')[0].split('<|return|>')[0].strip()
        return raw

    def generate_question(self, source_concept: str, target_concept: str, paths: List[Dict]) -> Optional[str]:
        paths_str = ', '.join([f"({p['start']}, {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""
        Create a {self.domain} exam question (similar to CCNA/CCNP or networking graduate-level exams)
        for advanced students that tests the relationship between '{source_concept}' and '{target_concept}'.
        The underlying relationship is: {paths_str}.

        The question should:
        1. Be in multiple choice format with exactly 4 options
        2. Be grounded in a concrete networking scenario — e.g., a packet trace, a misconfigured router,
           a protocol exchange, a network topology decision, or a troubleshooting situation
        3. Require multi-step reasoning that follows the relationship; do not make the answer
           obvious from the question stem alone
        4. Not directly state the relationship or the concept names from the path in the question stem
        5. Have exactly one clearly correct answer; the three distractors should be plausible but wrong

        Format your response exactly as:
        <Question>
        [Networking scenario and question]
        </Question>
        <Options>
        A. [Option]
        B. [Option]
        C. [Option]
        D. [Option]
        </Options>
        <Answer> [Correct Option Letter] </Answer>
        """
        return self._generate(prompt)

    def separate_question_and_answer(self, question: str) -> Tuple[str, str]:
        question_extracted = question.split('<Answer>')[0]
        answer = question.split('<Answer>')[1].split('</Answer>')[0].strip()
        # normalise "** B **" or "**B**" → "B"
        answer = answer.strip('* ').strip()
        return question_extracted, answer

    def quality_filtering(self, question: str) -> bool:
        required_tags = ['<Question>', '</Question>', '<Options>', '</Options>']
        if not all(tag in question for tag in required_tags):
            print("Required tags not present")
            return False
        if not all(opt in question for opt in ['A.', 'B.', 'C.', 'D.']):
            print("Options not present")
            return False
        for line in question.splitlines():
            for opt in ['A.', 'B.', 'C.', 'D.']:
                if all(ord(c) < 128 for c in line.strip()) and line.strip().startswith(opt) and len(line.strip()) < 5:
                    print("Short ASCII-only option")
                    return False
        prompt = f"""
        Check whether the answer options in this question are near-duplicates of each other.
        Respond with only 'Yes' or 'No'. 'Yes' if the options are distinct, 'No' if they are near-duplicates.
        Question: {question}
        """
        try:
            content = self._generate(prompt, max_tokens=10, temperature=0.0)
            if content and content.strip().lower().startswith('no'):
                print("Options are near duplicates")
                return False
        except Exception as e:
            print(f"Error checking question quality: {e}")
            return False
        return True

    def generate_thinking_trace(self, question: str, paths: List[Dict]) -> Optional[str]:
        paths_str = ', '.join([f"({p['start']}, {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""
        Generate a detailed step-by-step {self.domain} reasoning trace that arrives at the correct answer
        to this question: {question}

        Use the following relationship as the backbone of your reasoning: {paths_str}.

        The explanation should:
        1. Walk through the {self.domain} concepts step by step, referencing relevant protocols,
           layers, or mechanisms as appropriate.
        2. Use the provided relationship to motivate each reasoning step without explicitly
           quoting it — the logic should flow naturally from networking first principles.
        3. Eliminate each wrong option with a brief technical justification.
        4. Sound like a senior networking engineer explaining their thought process to a colleague.
        """
        return self._generate(prompt, max_tokens=2048)

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        return f"{question}\n<Explanation>\n{explanation}\n</Explanation>\n<Answer>:\n{answer}\n</Answer>"

    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> str:
        paths_str = ', '.join([f"({p['start']}, {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""You are a senior {self.domain} engineer and exam question reviewer.
        You are given a multiple-choice question, a reasoning explanation, and a selected answer.
        The question was generated from this knowledge graph path: {paths_str}.

        Evaluate the following:
        1. Is the selected answer technically correct from a {self.domain} standpoint?
        2. Does the reasoning explanation correctly justify that answer?
        3. Are the three distractors plausibly wrong (not trick questions or obviously absurd)?

        Respond in exactly this format — nothing else:
        Correct: [Yes/No]

        Question, explanation and answer:
        {question_answer_explanation}
        """
        try:
            content = self._generate(prompt, max_tokens=20, temperature=0.0)
        except Exception as e:
            print(f"Error in correctness filtering: {e}")
            return "error"

        match = re.search(r'Correct:\s*(Yes|No)', content, re.IGNORECASE)
        if match is None:
            return "error"
        return "Yes" if match.group(1).lower() == "yes" else "No"


def _shuffle_answer_options(question_text: str, correct_answer: str) -> Tuple[str, str]:
    """Randomly reassign A/B/C/D to the four options and return the updated text and new correct letter."""
    options_match = re.search(r'(<Options>)(.*?)(</Options>)', question_text, re.DOTALL)
    if not options_match:
        return question_text, correct_answer

    options_content = options_match.group(2)
    # Parse lines of the form "A. <text>" allowing multi-line option text
    option_lines = re.findall(r'([A-D])\.\s+(.+?)(?=\n[A-D]\.|$)', options_content, re.DOTALL)
    if len(option_lines) != 4:
        return question_text, correct_answer

    letters = ['A', 'B', 'C', 'D']
    orig_texts = {letter: text.strip() for letter, text in option_lines}
    correct = correct_answer.strip().upper()
    if correct not in orig_texts:
        return question_text, correct_answer

    # Shuffle by permuting indices so we can track which slot the correct answer lands in
    indices = list(range(4))
    random.shuffle(indices)
    correct_orig_idx = letters.index(correct)
    new_correct = letters[indices.index(correct_orig_idx)]

    shuffled_texts = [orig_texts[letters[i]] for i in indices]
    new_options_block = '\n' + '\n'.join(f'{letters[i]}. {shuffled_texts[i]}' for i in range(4)) + '\n'

    new_question_text = (
        question_text[:options_match.start(2)] +
        new_options_block +
        question_text[options_match.end(2):]
    )
    return new_question_text, new_correct


class QAGenerator:
    def __init__(self, api_key: str = None, model_name: str = 'openai/gpt-oss-120b',
                 domain: str = 'computer networking',
                 vocab_path: str = None, graph_path: str = None, categories_path: str = None):
        if api_key:
            os.environ['TINKER_API_KEY'] = api_key

        base_dir = os.path.dirname(__file__)
        self.llm = TinkerLLMBackend(model_name=model_name, domain=domain)
        self.generator = PathGenerator(
            vocab_path=vocab_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'vocab.txt'),
            graph_path=graph_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'networks.graph'),
            categories_path=categories_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'categories.json'),
        )

    def generate_questions(self, k_hops: int = 1, category: str = None) -> Dict:
        path = self.generator.generate_paths(category=category, k_hops=k_hops)
        question = self.llm.generate_question(
            source_concept=path['source_concept'],
            target_concept=path['target_concept'],
            paths=path['paths'],
        )
        question, answer = self.llm.separate_question_and_answer(question)
        question, answer = _shuffle_answer_options(question, answer)
        path['question'] = question
        path['answer'] = answer
        return path

    def quality_filtering(self, question: str) -> bool:
        return self.llm.quality_filtering(question)

    def generate_thinking_trace(self, question: str, paths: List[Dict]) -> Optional[str]:
        return self.llm.generate_thinking_trace(question, paths)

    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> str:
        return self.llm.correctness_filtering(question_answer_explanation, paths)

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        return self.llm.combine_question_and_thinking_trace_with_answer(question, explanation, answer)
