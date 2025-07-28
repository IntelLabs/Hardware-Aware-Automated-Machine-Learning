import json
import os
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict

from datasets import load_dataset


def is_single_image(images) -> bool:
    """Check if the input contains exactly one image."""
    return len(images) == 1


def is_image_size_valid(image, min_pixels: int = 3136, max_pixels: int = 12845056, factor: int = 28) -> bool:
    """Check if the image size is within the specified pixel range."""
    w, h = image.size
    if h < factor or w < factor:
        return False
    if min(h, w) == 0 or max(h, w) / min(h, w) > 200:
        return False
    num_pixels = w * h
    return min_pixels <= num_pixels <= max_pixels


def load_generic_image_text_dataset(
    dataset_name: str,
    subset: str = None,
    limit: int = None,
    id_prefix: str = ""
) -> Tuple[List[dict], Dict[str, object]]:
    """
    Generic loader for image-text datasets with unified logic.
    :param dataset_name: Name of the dataset.
    :param subset: Subset name (optional).
    :param limit: Limit the number of samples (optional).
    :param id_prefix: Prefix for image IDs.
    :return: Tuple of processed samples and image ID mapping.
    """
    if subset is not None:
        data_iter = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        data_iter = load_dataset(dataset_name, split="train", streaming=True)

    raw_data = []
    num_skipped = 0
    for example in tqdm(data_iter):
        if "images" in example:
            if not is_single_image(example["images"]):
                num_skipped += 1
                continue
            image = example["images"][0]
        else:
            assert "image" in example, f"Dataset {dataset_name} does not contain 'images' or 'image' field."
            image = example["image"]
            
        if not is_image_size_valid(image):
            num_skipped += 1
            continue
        raw_data.append(example)
        if limit is not None and len(raw_data) == limit:
            break

    print(f"There are {len(raw_data)} valid samples, and {num_skipped} samples are skipped.")

    tag = "<image>\n"
    processed_samples = []
    id_image_mapping = {}

    image_id = 0
    for example in tqdm(raw_data):
        image = example["images"][0] if "images" in example else example["image"]
        if subset is not None:
            id = f"{id_prefix}_{subset}_{image_id}"
        else:
            id = f"{id_prefix}_{image_id}"
        image_id += 1
        id_image_mapping[id] = image

        if dataset_name == "HuggingFaceH4/llava-instruct-mix-vsft":
            # LLaVA-Instruct-Mix-VSFT: messages is a list of dicts with 'content' and 'role'
            contents = example["messages"]
            for idx in range(len(contents) - 1):
                if contents[idx]["role"] == "user" and contents[idx + 1]["role"] == "assistant":
                    query_parts = []
                    for item in contents[idx]["content"]:
                        if item["type"] == "text":
                            query_parts.append(item["text"] if item["text"] is not None else "")
                    query = tag + "".join(query_parts)
                    answer_parts = []
                    for item in contents[idx + 1]["content"]:
                        if item["type"] == "text":
                            answer_parts.append(item["text"] if item["text"] is not None else "")
                    answer = "".join(answer_parts)
                    processed_item = {
                        "image_id": id,
                        "query": query,
                        "answer": answer
                    }
                    processed_samples.append(processed_item)
        elif dataset_name == "HuggingFaceM4/ChartQA":
            processed_item = {
                "image_id": id,
                "query": tag + example["query"],
                "answer": example["label"][0]
            }
            processed_samples.append(processed_item)
        else:
            for text in example["texts"]:
                processed_item = {
                    "image_id": id,
                    "query": tag + text["user"],
                    "answer": text["assistant"]
                }
                processed_samples.append(processed_item)

    return processed_samples, id_image_mapping


def load_llava_instruct_mix_vsft_dataset(limit: int = None) -> Tuple[List[dict], Dict[str, object]]:
    """
    Loader for HuggingFaceH4/llava-instruct-mix-vsft dataset.
    """
    return load_generic_image_text_dataset(
        dataset_name="HuggingFaceH4/llava-instruct-mix-vsft",
        limit=limit,
        id_prefix="llava_instruct_mix_vsft",
    )


def load_the_cauldron_dataset(subset: str = None, limit: int = None) -> Tuple[List[dict], Dict[str, object]]:
    """
    Loader for HuggingFaceM4/the_cauldron dataset.
    """
    return load_generic_image_text_dataset(
        dataset_name="HuggingFaceM4/the_cauldron",
        subset=subset,
        limit=limit,
        id_prefix="the_cauldron"
    )


def load_docmatix_dataset(limit: int = None) -> Tuple[List[dict], Dict[str, object]]:
    """
    Loader for HuggingFaceM4/Docmatix dataset.
    """
    return load_generic_image_text_dataset(
        dataset_name="HuggingFaceM4/Docmatix",
        subset="images",
        limit=limit,
        id_prefix="docmatix"
    )


def load_chartqa_dataset(limit: int = None) -> Tuple[List[dict], Dict[str, object]]:
    """
    Loader for HuggingFaceM4/ChartQA dataset.
    """
    return load_generic_image_text_dataset(
        dataset_name="HuggingFaceM4/ChartQA",
        limit=limit,
        id_prefix="chartqa"
    )


def load_data(datasets: List[str], limit: int = None, debug: bool = False) -> Tuple[List[str], Dict[int, str]]:
    all_samples = []
    all_id_image_mapping = {}
    idx_to_dataset = {}
    current_index = 0
    limit = 100 if debug else limit
    for dataset in datasets:    
        if "HuggingFaceM4/the_cauldron" in dataset:
            parts = dataset.split('/')
            if len(parts) == 2:
                samples, id_image_mapping = load_the_cauldron_dataset(limit=limit)
            elif len(parts) == 3:
                subset = parts[2]
                samples, id_image_mapping = load_the_cauldron_dataset(subset=subset, limit=limit)
        elif dataset == "HuggingFaceM4/Docmatix":
            if limit is None:
                limit = 100000  # Default limit for Docmatix
            samples, id_image_mapping = load_docmatix_dataset(limit=limit)
        elif dataset == "HuggingFaceH4/llava-instruct-mix-vsft":
            samples, id_image_mapping = load_llava_instruct_mix_vsft_dataset(limit=limit)
        elif dataset == "HuggingFaceM4/ChartQA":
            samples, id_image_mapping = load_chartqa_dataset(limit=limit)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        for i in range(len(samples)):
            idx_to_dataset[current_index + i] = dataset
        current_index += len(samples)

        all_samples.extend(samples)
        for k, v in id_image_mapping.items():
            assert k not in all_id_image_mapping
            all_id_image_mapping[k] = v
        
        info = {
            "dataset_name": dataset,
            "num_of_images": len(id_image_mapping),
            "num_of_samples": len(samples),
        }
        print(json.dumps(info, indent=4))
    
    print(f"Number of samples: {len(all_samples)}, number of images: {len(all_id_image_mapping)}")

    return all_samples, all_id_image_mapping, idx_to_dataset


def load_knowledge_base(database_paths: List[str]) -> Tuple[List[str], torch.Tensor]:
    all_samples = []
    all_embs = []
    for database_path in database_paths:
        with open(os.path.join(database_path, "data.json"), "r") as f:
            samples = json.load(f)
        embs = torch.load(os.path.join(database_path, "embeddings.pt"))
        all_samples.extend(samples)
        all_embs.append(embs.cpu())
        print(f"Loaded {len(samples)} samples from {database_path}, embeddings shape: {embs.shape}")
    if all_embs:
        merged_embs = torch.cat(all_embs, dim=0)
    else:
        merged_embs = torch.empty(0)
    return all_samples, merged_embs
