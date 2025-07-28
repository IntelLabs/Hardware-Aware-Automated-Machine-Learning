# This script is adapted from the implementation found at https://github.com/QwenLM/Qwen2.5-VL
import copy
from dataclasses import dataclass
from typing import Dict, Sequence, List
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizer

from data.rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100


def preprocess_qwen_2_visual(
    sources,
    tokenizer: PreTrainedTokenizer,
    grid_thw_image: List = [],
) -> Dict:
    assert len(sources) == 1
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        # user
        role = "user"
        content = source["query"]

        if "<image>" in content:
            parts = content.split("<image>")
            new_parts = []
            for i in range(len(parts) - 1):
                new_parts.append(parts[i])
                replacement = (
                    "<|vision_start|>"
                    + f"<|image_pad|>"
                    * grid_thw_image[visual_replicate_index_image]
                    + "<|vision_end|>"
                )
                new_parts.append(replacement)
                visual_replicate_index_image += 1
            new_parts.append(parts[-1])
            content = "".join(new_parts)
        
        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        target += [IGNORE_INDEX] * len(encode_id)

        # assistant
        role = "assistant"
        content = source["answer"]
        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        target_mask = encode_id.copy()
        target_mask[:3] = [IGNORE_INDEX] * 3
        target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
        samples: List, 
        model_type: str,
        tokenizer: PreTrainedTokenizer, 
        processor: AutoProcessor,
        max_pixels: int = 576*28*28,
        min_pixels: int = 16*28*28,
    ):
        super(LazySupervisedDataset, self).__init__()
        if model_type == "qwen2_5_vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        self.tokenizer = tokenizer
        self.image_processor = processor.image_processor
        self.image_processor.max_pixels = max_pixels
        self.image_processor.min_pixels = min_pixels
        self.image_processor.size["longest_edge"] = max_pixels
        self.image_processor.size["shortest_edge"] = min_pixels

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self._get_item(i)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.samples[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        grid_thw = None
        second_per_grid_ts = None

        if "images" in sources[0]:
            image_file = self.samples[i]["images"]
            if len(image_file) > 1:
                results = [self.process_image_unified(file) for file in image_file]
                image, grid_thw = zip(*results)
            else:
                image_file = image_file[0]
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
    
        chat_sources = copy.deepcopy(sources)
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "images" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        if "images" in self.samples[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["position_ids"] = position_ids
        return batch
