import os
import argparse
import json
from tqdm import tqdm
import torch

from models.embedder import load_clip_vit_embedder, get_multimodal_embedding
from utils import load_data


def get_file_size(file_path):
    return os.path.getsize(file_path)


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += get_file_size(fp)
    return total_size


def convert_size(size_bytes):
    return size_bytes / 1024  # Convert bytes to KB


def save_info_json(save_dir, num_data_samples, num_images, image_sizes, args):
    info = {}
    info["samples_count"] = num_data_samples
    info["images_count"] = num_images
    
    image_folder = os.path.join(save_dir, "images")
    data_json_path = os.path.join(save_dir, "data.json")
    embeddings_path = os.path.join(save_dir, "embeddings.pt")
    
    info["images_folder_size_kb"] = convert_size(get_folder_size(image_folder))
    info["data_json_size_kb"] = convert_size(get_file_size(data_json_path))
    info["embeddings_size_kb"] = convert_size(get_file_size(embeddings_path))
    
    info["average_image_size_kb"] = sum(image_sizes) / len(image_sizes) if image_sizes else 0
    info["max_image_size_kb"] = max(image_sizes) if image_sizes else 0
    info["min_image_size_kb"] = min(image_sizes) if image_sizes else 0
    
    info["args"] = vars(args)
    
    with open(os.path.join(save_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate knowledge base for specified datasets.")
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='List of datasets to load in knowledge base.',
        default=[
            "HuggingFaceM4/Docmatix",
            "HuggingFaceH4/llava-instruct-mix-vsft",
            "HuggingFaceM4/ChartQA",
            "HuggingFaceM4/the_cauldron/ai2d",
            "HuggingFaceM4/the_cauldron/visual7w",
            "HuggingFaceM4/the_cauldron/aokvqa",
            "HuggingFaceM4/the_cauldron/hateful_memes",
            "HuggingFaceM4/the_cauldron/vqarad",
            "HuggingFaceM4/the_cauldron/docvqa",
            "HuggingFaceM4/the_cauldron/textcaps",
            "HuggingFaceM4/the_cauldron/textvqa",
            "HuggingFaceM4/the_cauldron/st_vqa",
            "HuggingFaceM4/the_cauldron/visualmrc",
            "HuggingFaceM4/the_cauldron/iam",
            "HuggingFaceM4/the_cauldron/infographic_vqa",
            "HuggingFaceM4/the_cauldron/diagram_image_to_text",
            "HuggingFaceM4/the_cauldron/chartqa",
            "HuggingFaceM4/the_cauldron/tat_qa",
            "HuggingFaceM4/the_cauldron/hitab",
            "HuggingFaceM4/the_cauldron/multihiertt",
            "HuggingFaceM4/the_cauldron/finqa",
            "HuggingFaceM4/the_cauldron/iconqa",
            "HuggingFaceM4/the_cauldron/intergps",
            "HuggingFaceM4/the_cauldron/tqa",
            "HuggingFaceM4/the_cauldron/scienceqa",
        ],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="knowledge_base",
        help="Path to save the cached embeddings and texts of the database. "
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit for the number of samples."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode."
    )
    args = parser.parse_args()
    
    datasets = args.datasets
    save_dir = args.save_dir
    limit = args.limit
    debug = args.debug

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load and concatenate datasets
    data_samples, id_image_mapping, idx_to_dataset = load_data(datasets, limit=limit, debug=debug)

    with open(os.path.join(save_dir, "idx_to_dataset.json"), "w") as f:
        json.dump(idx_to_dataset, f, indent=4)

    image_folder = os.path.join(save_dir, "images")
    os.makedirs(image_folder, exist_ok=True)
    
    # Save images
    for id, image in tqdm(id_image_mapping.items(), desc="Saving images"):
        image_path = os.path.join(image_folder, f"{id}.jpg")
        # Convert RGBA or P to RGB if necessary
        if image.mode in ['RGBA', 'P']:
            image = image.convert('RGB')
        image.save(image_path)
    
    # Save samples
    for sample in data_samples:
        sample["image_path"] = os.path.join(image_folder, f"{sample['image_id']}.jpg")
    with open(os.path.join(save_dir, "data.json"), "w") as f:
        # Remove "image_id" from each sample before saving
        samples_to_save = [{k: v for k, v in sample.items() if k != "image_id"} for sample in data_samples]
        # Change "image_paths" to "images"
        for sample in samples_to_save:
            sample["images"] = [sample.pop("image_path")]
        json.dump(samples_to_save, f, indent=4)

    num_data_samples = len(data_samples)
    num_images = len(id_image_mapping)
    image_sizes = [convert_size(get_file_size(os.path.join(image_folder, f"{id}.jpg"))) for id in id_image_mapping]

    # Embedding (for each data point) - mini-batch processing for efficiency
    embedder = load_clip_vit_embedder()
    batch_size = 128
    embeddings = []
    num_samples = len(data_samples)
    for start in tqdm(range(0, num_samples, batch_size), desc="Generating embeddings (batched)"):
        end = min(start + batch_size, num_samples)
        batch_samples = data_samples[start:end]
        # Prepare batch queries and images
        batch_queries = [sample["query"].replace("<image>\n", "") for sample in batch_samples]
        batch_images = [id_image_mapping[sample["image_id"]] for sample in batch_samples]

        # Generate embeddings in batch
        text_embeddings = embedder(batch_queries)    # Assumes embedder supports list[str]
        image_embeddings = embedder(batch_images)    # Assumes embedder supports list[Image]

        # Combine multimodal embeddings for each sample in the batch
        for t_emb, i_emb in zip(text_embeddings, image_embeddings):
            embedding = get_multimodal_embedding(t_emb, i_emb)
            embeddings.append(embedding)
        
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, os.path.join(save_dir, "embeddings.pt"))

    # Save info.json
    save_info_json(save_dir, num_data_samples, num_images, image_sizes, args)


if __name__ == "__main__":
    main()
