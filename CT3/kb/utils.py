import json
import os
import torch
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from itertools import zip_longest
from tqdm import tqdm


def get_file_size(file_path):
    return os.path.getsize(file_path)


def convert_size(size_bytes):
    return size_bytes / 1024  # Convert bytes to KB


def save_log_file(save_dir, num_data_samples, args):
    info = {}
    info["samples_count"] = num_data_samples
    data_json_path = os.path.join(save_dir, "texts.jsonl")
    embeddings_path = os.path.join(save_dir, "embeddings.pt")
    
    info["data_json_size_kb"] = convert_size(get_file_size(data_json_path))
    info["embeddings_size_kb"] = convert_size(get_file_size(embeddings_path))
    
    info["args"] = vars(args)
    
    with open(os.path.join(save_dir, "log.json"), "w") as f:
        json.dump(info, f, indent=4)


def load_coqa_dataset(limit: int = None) -> List[str]:

    def doc_to_text(doc: Dict[str, Any]) -> str:
        # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
        # and a question qi, the task is to predict the answer ai
        doc_text = doc["story"] + "\n\n"
        for q, a in zip_longest(
            doc["questions"]["input_text"], doc["answers"]["input_text"][:-1]
        ):  # omit target answer ai
            question = f"Q: {q}\n\n"
            answer = f"A: {a}\n\n" if a is not None else "A:"
            doc_text += question + answer
        return doc_text

    def doc_to_target(doc: Dict[str, Any]) -> str:
        turn_id = len(doc["questions"]["input_text"])
        answer_forturn = doc["answers"]["input_text"][turn_id - 1]
        return answer_forturn
    
    data = load_dataset("EleutherAI/coqa", split="train")
    samples = []
    for item in tqdm(data, desc="Processing COQA dataset"):
        prompt = doc_to_text(item)
        answer = doc_to_target(item)
        prompt = prompt + " " + answer
        samples.append(prompt)
        if limit is not None and len(samples) == limit:
            break

    return samples


def load_metamathqa_dataset(limit: int = None):
    data = load_dataset("meta-math/MetaMathQA", split="train")
    samples = []
    for item in tqdm(data, desc="Processing MetaMathQA dataset"):
        query = item["query"]
        response = item["response"]
        prompt = query + " " + response
        samples.append(prompt)
        if limit is not None and len(samples) == limit:
            break
    return samples


def load_orca_math_dataset(limit: int = None):
    data = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    samples = []
    for item in tqdm(data, desc="Processing Orca Math dataset"):
        query = item["question"]
        response = item["answer"]
        prompt = query + " " + response
        samples.append(prompt)
        if limit is not None and len(samples) == limit:
            break
    return samples


def load_math_50k_dataset(limit: int = None) -> List[str]:
    samples = []
    with open("math_50k.json", "r") as f:
        data = json.load(f)
    for item in tqdm(data, desc="Processing Math 50k dataset"):
        text = item["instruction"] + item["input"] + item["output"]
        samples.append(text)
        if limit is not None and len(samples) == limit:
            break
    return samples


def load_stack_python_dataset(limit: int = None):
    samples = []
    # dataset streaming (will only download the data as needed)
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", streaming=True, split="train")
    ds = ds.shuffle(buffer_size=10000, seed=42)
    # take the first 600000 samples from the shuffled dataset
    sampled_ds = ds.take(600000)
    for sample in tqdm(sampled_ds, desc="Processing Stack Python dataset", total=600000):
        samples.append(sample["content"])
        if limit is not None and len(samples) == limit:
            break

    return samples


def load_mbpp_dataset(limit: int = None) -> List[str]:
    data = load_dataset("mbpp", split="train")
    samples = []
    for item in tqdm(data, desc="Processing MBPP dataset"):
        description = item["text"]
        test_example = item["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        samples.append(prompt)
        if limit is not None and len(samples) == limit:
            break

    return samples


def load_data(datasets: List[str], debug: bool = False) -> Tuple[List[str], Dict[int, str]]:
    all_samples = []
    idx_to_dataset = {}
    current_index = 0
    
    limit = 100 if debug else None
    for dataset in datasets:
        print(f"Loading dataset: {dataset}")
        if dataset == "EleutherAI/coqa":
            samples = load_coqa_dataset(limit=limit)
        elif dataset == "meta-math/MetaMathQA":
            samples = load_metamathqa_dataset(limit=limit)
        elif dataset == "microsoft/orca-math-word-problems-200k":
            samples = load_orca_math_dataset(limit=limit)
        elif dataset == "math_50k":
            samples = load_math_50k_dataset(limit=limit)
        elif dataset == "bigcode/the-stack-python":
            samples = load_stack_python_dataset(limit=limit)
        elif dataset == "mbpp":
            samples = load_mbpp_dataset(limit=limit)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        print(f"Loaded {len(samples)} samples from {dataset}")
        for i, _ in enumerate(samples):
            idx_to_dataset[current_index + i] = dataset

        current_index += len(samples)
        all_samples.extend(samples)

    print(f"Total samples loaded: {len(all_samples)}")
    return all_samples, idx_to_dataset


def load_knowledge_base(database_path: str) -> Tuple[List[str], torch.Tensor]:
    texts = []
    with open(os.path.join(database_path, "texts.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)
            texts.append(text)

    embs = torch.load(os.path.join(database_path, "embeddings.pt"))
    return texts, embs.cpu()
