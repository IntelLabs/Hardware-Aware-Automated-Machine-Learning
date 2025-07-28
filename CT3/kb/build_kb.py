import os
import argparse
import json
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer

from models.embedder import Embedder, EMBEDDING_MODEL_PATH
from utils import load_data, save_log_file


def main():
    parser = argparse.ArgumentParser(description="Generate knowledge base for specified datasets.")
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=['EleutherAI/coqa', 'meta-math/MetaMathQA', 'microsoft/orca-math-word-problems-200k', 'math_50k', 'bigcode/the-stack-python', 'mbpp'],
        required=True,
        help='List of datasets to load in knowledge base. '
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="knowledge_base",
        help="Path to save the cached embeddings and texts of the database. "
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    args = parser.parse_args()
    
    datasets = args.datasets
    save_dir = args.save_dir
    debug = args.debug

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load the tokenizer and model for embedding
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH, device_map={"": 0}, trust_remote_code=True)
    embedder = Embedder(model, tokenizer)

    # Load and concatenate datasets
    data_samples, idx_to_dataset = load_data(datasets, debug=debug)

    with open(os.path.join(save_dir, "idx_to_dataset.json"), "w") as f:
        json.dump(idx_to_dataset, f, indent=4)
    
    embeddings = []
    texts = []
    for sample in tqdm(data_samples, desc="Generating Embeddings"):
        embedding = embedder(sample)
        embeddings.append(embedding)
        texts.append(sample)
    
    # Save the texts to a file using JSON format
    with open(os.path.join(save_dir, "texts.jsonl"), "w", encoding="utf-8") as f:
        for text in texts:
            json.dump(text, f)
            f.write("\n")
    
    # Concatenate all embeddings and save to the specified directory
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, os.path.join(save_dir, "embeddings.pt"))
    
    # Save log.json
    save_log_file(save_dir, len(texts), args)


if __name__ == "__main__":
    main()
