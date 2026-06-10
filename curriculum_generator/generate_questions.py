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
from typing import List, Dict, Tuple
from google import genai
from typing import Optional
import os
import re
import asyncio
import tinker
from tinker import types as tinker_types

class PathGenerator:
    def __init__(self, vocab_path: str, graph_path: str, icd10_categories_path: str, vocab_freq_path: str = None):
        # Load resources
        self.vocab = self._load_vocab(vocab_path)
        self.concept2id = {w: i for i, w in enumerate(self.vocab)}
        self.graph = self._load_graph(graph_path)
        self.icd10_categories = self._load_icd10_categories(icd10_categories_path)
        self.vocab_freq = {vocab:0 for vocab in self.vocab}
        if vocab_freq_path is not None and os.path.exists(vocab_freq_path):
            self.vocab_freq = self.__update_vocab_freq(vocab_freq_path)
            print("Updated vocab freq")
        self.merged_relations = [
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

    def __update_vocab_freq(self, path: str) -> Dict:
        with open(path, 'r') as f:
            vocab_freq = json.load(f)
        for vocab in vocab_freq:
            self.vocab_freq[vocab] = vocab_freq[vocab]
        return self.vocab_freq

    def _load_vocab(self, path: str) -> List[str]:
        with open(path, 'r') as f:
            return f.read().splitlines()
    

    def _load_graph(self, path: str) -> nx.Graph:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_icd10_categories(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def _get_k_hop_neighbors(self, start_node: int, k: int) -> Tuple[List[Tuple[int, int, str]], bool]:
        """Get k-hop neighbors and their relationships from start_node.
        
        Returns:
            Tuple containing:
            - List of (start_node, end_node, relation) tuples
            - Boolean indicating if a valid path was found
        """
        paths = []
        all_nodes = set([start_node])
        current_node = start_node
        
        for hop in range(k):
            neighbors = list(self.graph.neighbors(current_node))
            # Filter out neighbors with unwanted relations
            filtered_neighbors = []
            for n in neighbors:
                rel = self.graph[current_node][n][0]['rel']
                if rel >= len(self.merged_relations):
                    rel = rel - len(self.merged_relations)
                relation = self.merged_relations[rel]
                if relation not in ['is_a', 'part_of', 'has_part', 'contains']:
                    filtered_neighbors.append(n)
            neighbors = filtered_neighbors
            available_neighbors = [n for n in neighbors if n not in all_nodes]
            
            if not available_neighbors:
                return [], False  # No valid neighbors left, signal failure
                
            neighbor = random.choice(available_neighbors)
            all_nodes.add(neighbor)
            rel = self.graph[current_node][neighbor][0]['rel']
            if rel >= len(self.merged_relations):
                rel = rel - len(self.merged_relations)
            paths.append((current_node, neighbor, self.merged_relations[rel]))
            current_node = neighbor
            
        return paths, True

    def generate_paths(self, category, k_hops: int = 1) -> Dict:
        """Generate paths from a specific ICD10 category using k-hop neighbors."""
    
        if category is not None:
            concepts_in_category = self.icd10_categories[category]
        
        # Keep trying until we find a valid path
        max_attempts = 10
        for _ in range(max_attempts):
            
            if category is not None:
                sampled_concept = random.sample(concepts_in_category, 1)[0]
            else:
                # Sample a concept from vocab inversely proportional to its frequency
                # Get all concepts with their frequencies
                concepts = list(self.vocab_freq.keys())
                # Calculate inverse frequencies (add small epsilon to avoid division by zero)
                inverse_freqs = [1.0 / (freq + 1e-10) for freq in self.vocab_freq.values()]
                # Normalize to create a probability distribution
                total = sum(inverse_freqs)
                probs = [freq / total for freq in inverse_freqs]
                # Sample a concept based on inverse frequency distribution
                sampled_concept = random.choices(concepts, weights=probs, k=1)[0]
                
            if sampled_concept not in self.concept2id:
                continue
            
            concept_id = self.concept2id[sampled_concept]
            paths, success = self._get_k_hop_neighbors(concept_id, k_hops)
            
            if success:
                return {
                    "source_concept": sampled_concept,
                    "paths": [
                        {
                            "start": self.vocab[start],
                            "end": self.vocab[end],
                            "relation": relation
                        }
                        for start, end, relation in paths
                    ],
                    'target_concept': self.vocab[paths[-1][1]],
                }
        
        raise ValueError(f"Could not find valid path after {max_attempts} attempts")
    
class GeminiLLMBackend:
    def __init__(self, model_name_question: str = 'gemini-2.0-flash', model_name_explanation: str = 'gemini-2.5-flash-preview-05-20'):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        self.client =genai.Client(api_key=self.api_key)
        self.model_question = model_name_question
        self.model_explanation = model_name_explanation

        
    def generate_question(self, source_concept: str, target_concept: str, paths: List[Dict]) -> Optional[str]:
        """
        Generate a medical examination question using LLM based on two related medical concepts.
        
        Args:
            source_concept: The source medical concept
            target_concept: The target medical concept
            path: The path between the concepts
            
        Returns:
            str: A generated medical examination question
        """
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""
        Create a medical examination question (like those found in medical board exams) for advanced medical students that tests the relationship between 
        '{source_concept}' and '{target_concept}'. \n The relationship is: {paths_str}. 
        
        The question should:
        1. Be in multiple choice format (4 options)
        2. Require clinical reasoning along the relationship
        3. Include a brief clinical vignette
        4. Not directly mention the relationship in the question stem
        5. Have one clearly correct answer

        Format:
        <Question>
        [Clinical Vignette]
        </Question>
        <Options>
        A. [Option]
        B. [Option]
        C. [Option]
        D. [Option]
        </Options>
        <Answer> [Correct Option Letter] </Answer>
        """
        try:
            response = self.client.models.generate_content(
                model = self.model_question,
                contents = prompt
            )
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            print(f"Error generating question: {e}")
            return None
        
    def separate_question_and_answer(self, question: str) -> Tuple[str, str]:
        """
        Separate the question and the answer from the question string.
        """
        question_extracted = question.split('<Answer>')[0]
        answer = question.split('<Answer>')[1].split('</Answer>')[0].strip()

        return question_extracted, answer
    
    def quality_filtering(self, question: str) -> bool:
        """
        Checks for question quality and formatting errors
        """
        # Check for required tags and formatting
        required_tags = ['<Question>', '</Question>', '<Options>', '</Options>']
        if not all(tag in question for tag in required_tags):
            print("Required tags not present")
            return False
        # Check for all options present and formatted
        options = ['A.', 'B.', 'C.', 'D.']
        if not all(opt in question for opt in options):
            print("Options not present")
            return False
        # Optionally, check for suspiciously short ASCII-only options
        for line in question.splitlines():
            for opt in ['A.', 'B.', 'C.', 'D.']:
                if all(ord(c) < 128 for c in line.strip()) and line.strip().startswith(opt) and len(line.strip()) < 5:
                    print("Short ASCII-only option")
                    return False
        # Check for near duplicates
        prompt = f"""
        You will be given a question and you will need to check the question to ensure the options are not near duplicates of each other.
        Only respond with: 'Yes' or 'No', nothing else. "Yes' if the options are not near duplicates of each other, 'No' otherwise.
        Check the question: {question}
        """
        try:
            response = self.client.models.generate_content(
                model = self.model_question,
                contents = prompt
            )
            content = response.candidates[0].content.parts[0].text
            content = content.strip().lower()
            if content == 'no':
                print("Options are near duplicates")
                return False
        except Exception as e:
            print(f"Error checking question quality: {e}")
            return False
        
        return True
        
    def generate_thinking_trace(self, question: str, paths: List[Dict]) -> Optional[str]:
        """
        Generate a step by step thinking trace of the correct answer to the question using the provided paths.
        """
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""
        Generate a detailed thinking trace of the answer to the question: {question} \n Use the following context to generate the thinking trace: {paths_str}.

        The explanation should be:
        1. Detailed and include all the steps leading to the correct answer.
        2. You are to use the provided context to explain the relationship between the concepts.
        3. Strictly do not mention that you are using the provided context to generate the explanation.
        4. Your explanation should sound like a medical student explaining to a peer.
        """
        try:
            response = self.client.models.generate_content(
                model = self.model_explanation,
                contents = prompt,
            )
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            print(f"Error generating COT explanation: {e}")
            return None

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        """
        Combine the question and the thinking trace into a single string.
        """
        return f"{question}\n<Explanation>\n{explanation}\n</Explanation>\n<Answer>:\n{answer}\n</Answer>"
    
    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        
        paths_str = ','.join([f"({path['start']} , {path['relation']}, {path['end']})" for path in paths])
        prompt = f"""You are a medical examiner. You are given a medical question and an explanation with an answer. The question and answer are formatted as follows:
        <Question>: [Clinical Vignette] </Question>
        <Options>
        A. [Option]
        B. [Option]
        C. [Option]
        D. [Option]
        </Options>
        <Explanation>: [Explanation] </Explanation>
        <Answer>: [Correct Option Letter] </Answer>

        1. Judge whether the question and answer are logically correct and medically accurate, and follow the source. If there is an explanation, also judge the explanation along with the answer and just evaluate the correctness of the answer otherwise.
        2. Respond with only "Yes" or "No".
        Format your response exactly like this:
        Correct: [Yes/No]

        Question and Answer: {question_answer_explanation}
        Source: {paths_str}
        """
    
        response = self.client.models.generate_content(
            model = self.model_explanation,
            contents = prompt
        )   
        try:
            response = self.client.models.generate_content(
                model = self.model_question,
                contents = prompt
            )
            content = response.candidates[0].content.parts[0].text
        except Exception as e:
            print(f"Error correcting: {e}")
            return None
        
        correct_match = re.search(r'Correct:\s*(Yes|No)', content, re.IGNORECASE)
        if correct_match is None:
            correct_str = "error"
        elif correct_match.group(1).lower() == "yes":
            correct_str = "Yes"
        elif correct_match.group(1).lower() == "no":
            correct_str = "No"
        else:
            correct_str = "error"
        
        return correct_str


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

        return asyncio.run(_run())

    def generate_question(self, source_concept: str, target_concept: str, paths: List[Dict]) -> Optional[str]:
        paths_str = ', '.join([f"({p['start']} , {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""
        Create a {self.domain} exam question for advanced students that tests the relationship between
        '{source_concept}' and '{target_concept}'. The relationship is: {paths_str}.

        The question should:
        1. Be in multiple choice format (4 options)
        2. Require reasoning along the relationship
        3. Include a brief scenario or context
        4. Not directly mention the relationship in the question stem
        5. Have one clearly correct answer

        Format:
        <Question>
        [Scenario or context]
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
        return question_extracted, answer

    def quality_filtering(self, question: str) -> bool:
        required_tags = ['<Question>', '</Question>', '<Options>', '</Options>']
        if not all(tag in question for tag in required_tags):
            print("Required tags not present")
            return False
        options = ['A.', 'B.', 'C.', 'D.']
        if not all(opt in question for opt in options):
            print("Options not present")
            return False
        for line in question.splitlines():
            for opt in ['A.', 'B.', 'C.', 'D.']:
                if all(ord(c) < 128 for c in line.strip()) and line.strip().startswith(opt) and len(line.strip()) < 5:
                    print("Short ASCII-only option")
                    return False
        prompt = f"""
        You will be given a question and you will need to check that the options are not near duplicates of each other.
        Only respond with: 'Yes' or 'No', nothing else. 'Yes' if the options are not near duplicates, 'No' otherwise.
        Check the question: {question}
        """
        try:
            content = self._generate(prompt, max_tokens=10, temperature=0.0)
            if content and content.strip().lower() == 'no':
                print("Options are near duplicates")
                return False
        except Exception as e:
            print(f"Error checking question quality: {e}")
            return False
        return True

    def generate_thinking_trace(self, question: str, paths: List[Dict]) -> Optional[str]:
        paths_str = ', '.join([f"({p['start']} , {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""
        Generate a detailed thinking trace for the answer to the following question: {question}
        Use this context to guide your reasoning: {paths_str}.

        The explanation should:
        1. Be detailed and include all steps leading to the correct answer.
        2. Use the provided context to explain the relationship between concepts.
        3. Not mention that you are using provided context.
        4. Sound like a knowledgeable student explaining to a peer.
        """
        return self._generate(prompt, max_tokens=2048)

    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        return f"{question}\n<Explanation>\n{explanation}\n</Explanation>\n<Answer>:\n{answer}\n</Answer>"

    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        paths_str = ', '.join([f"({p['start']} , {p['relation']}, {p['end']})" for p in paths])
        prompt = f"""You are an expert examiner in {self.domain}. You are given a question and an explanation with an answer formatted as:
        <Question>: [Scenario] </Question>
        <Options> A. ... B. ... C. ... D. ... </Options>
        <Explanation>: [Explanation] </Explanation>
        <Answer>: [Correct Option Letter] </Answer>

        Judge whether the question and answer are logically correct and accurate given the source.
        Respond with only "Yes" or "No" in this exact format:
        Correct: [Yes/No]

        Question and Answer: {question_answer_explanation}
        Source: {paths_str}
        """
        try:
            content = self._generate(prompt, max_tokens=20, temperature=0.0)
        except Exception as e:
            print(f"Error in correctness filtering: {e}")
            return None

        correct_match = re.search(r'Correct:\s*(Yes|No)', content, re.IGNORECASE)
        if correct_match is None:
            return "error"
        return "Yes" if correct_match.group(1).lower() == "yes" else "No"


class QAGenerator:
    def __init__(self, api_key: str = None, backend: str = 'tinker',
                 model_name: str = 'Qwen/Qwen3-32B-Instruct-2507',
                 domain: str = 'computer networking',
                 vocab_path: str = None, graph_path: str = None, categories_path: str = None):
        base_dir = os.path.dirname(__file__)

        if backend == 'tinker':
            if api_key:
                os.environ['TINKER_API_KEY'] = api_key
            self.llm = TinkerLLMBackend(model_name=model_name, domain=domain)
        else:
            if api_key is None:
                raise ValueError("GOOGLE_API_KEY is required for gemini backend")
            os.environ['GOOGLE_API_KEY'] = api_key
            self.llm = GeminiLLMBackend()

        self.generator = PathGenerator(
            vocab_path=vocab_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'vocab.txt'),
            graph_path=graph_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'networks.graph'),
            icd10_categories_path=categories_path or os.path.join(base_dir, 'data_kg', 'networks_kg', 'categories.json')
        )

    def generate_questions(self, k_hops: int = 1, category: str = None) -> List[Dict]:
        path = self.generator.generate_paths(
            category=category,\
            k_hops=k_hops
        )
       
        question = self.llm.generate_question(
            source_concept=path['source_concept'],\
            target_concept=path['target_concept'],\
            paths=path['paths']
        )
        question, answer = self.llm.separate_question_and_answer(question)
        path['question'] = question
        path['answer'] = answer

        return path
    
    def quality_filtering(self, question: str) -> bool:
        """
        Checks for question quality and formatting errors
        """
        return self.llm.quality_filtering(question)

    def generate_thinking_trace(self, question: str, paths: List[Dict]) -> Optional[str]:
        """
        Generates a thinking trace for the question
        """
        return self.llm.generate_thinking_trace(question, paths)
    
    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        """
        Checks for correctness of the question and answer
        """
        return self.llm.correctness_filtering(question_answer_explanation, paths)
    
    def combine_question_and_thinking_trace_with_answer(self, question: str, explanation: str, answer: str) -> str:
        """
        Combines the question and the thinking trace into a single string
        """
        return self.llm.combine_question_and_thinking_trace_with_answer(question, explanation, answer)
    
    def correctness_filtering(self, question_answer_explanation: str, paths: List[Dict]) -> bool:
        """
        Checks for correctness of the question and answer
        """
        return self.llm.correctness_filtering(question_answer_explanation, paths)


def main():
    # generate one question
    gemini_gym = QAGenerator()
    # Generate with Gemini
    print("Generating with Gemini...")
    question = gemini_gym.generate_questions(
        category='Certain infectious and parasitic diseases',
        k_hops=2
    )
    print(question['question'])
    quality = gemini_gym.quality_filtering(question['question'])
    if not quality:
        print("Quality filtering failed")
        return
    print("Quality filtering passed")
    explanation = gemini_gym.generate_thinking_trace(
        question=question['question'],
        paths=question['paths']
    )
    combined = gemini_gym.combine_question_and_thinking_trace_with_answer(
        question=question['question'],
        explanation=explanation,
        answer=question['answer']
    )
    correctness = gemini_gym.correctness_filtering(combined, question['paths'])
    if not correctness:
        print("Correctness filtering failed")
        return
    print("Correctness filtering passed")
    print(combined)
    

if __name__ == "__main__":
    main()