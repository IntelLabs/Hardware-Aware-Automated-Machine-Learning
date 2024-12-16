# From https://github.com/MrGGLS/BlockPruner/blob/main/utils.py
# Modified from https://github.com/microsoft/TransformerCompression
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import time
from typing import TypeVar, Union, Dict

import datasets
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import PreTrainedTokenizerBase

from lm_eval import evaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = TypeVar('T')


def map_tensors(obj: T, device: Union[torch.device, str, None] = None, dtype: Union[torch.dtype, None] = None) -> T:
    """Recursively map tensors to device and dtype."""
    if isinstance(obj, torch.Tensor):
        if device is not None:
            obj = obj.to(device=device)
        if dtype is not None:
            obj = obj.to(dtype=dtype)
        return obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device, dtype) for x in obj)
    elif isinstance(obj, dict):
        return {k: map_tensors(v, device, dtype) for k, v in obj.items()}  # type: ignore
    else:
        return obj


@torch.no_grad()
def evaluate_ppl(
    model,
    dataloader,
    pad_token_id=None,
    silence=True
) -> float:
    """
    Evaluate the model's perplexity on the test set using batch processing.
    It is expected that model is already on the correct device.
    """
    sync_gpus()

    start_time = time.time()

    model.eval()

    if pad_token_id:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_id)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    nlls = []

    if not silence:
        logging.info("Evaluating perplexity...")
    for batch in dataloader:
        if not silence:
            logging.debug(f"Evaluating batch {len(nlls)}")
        batch = map_tensors(batch, device)
        logits = model(**batch).logits

        # shift outputs and labels autoregressively.
        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        # CrossEntropyLoss demands data dimension is dimension 1.
        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()

        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()

    elapsed = time.time() - start_time
    if not silence:
        logging.info(
            "Time spent on evaluation: %s",
            time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
        )

    return ppl.item()


def sync_gpus() -> None:
    """Sync all GPUs to make sure all operations are finished, needed for correct benchmarking of latency/throughput."""
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


def get_dataset(name: str) -> datasets.DatasetDict:
    """
    Get the dataset from the HuggingFace datasets library.

    Args:
        name: The name of the HuggingFace dataset to load. Must be one of "wikitext2", "ptb", "c4" or "alpaca".

    Returns:
        The dataset.
    """
    logging.info(f"Loading dataset: {name}")

    ds_properties = {
        "wikitext2": {"path": "Salesforce/wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "allenai--c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ['url', 'timestamp'],
        },
        "alpaca": {"path": "tatsu-lab/alpaca", "cols_to_remove": ['input', 'output', 'instruction']},
    }

    if name not in ds_properties:
        raise NotImplementedError("The provided dataset is not supported")

    properties = ds_properties[name]
    ds = datasets.load_dataset(
        properties["path"], name=properties.get("config_name"), data_files=properties.get("data_files")
    )

    if "cols_to_remove" in properties:
        ds = ds.remove_columns(properties["cols_to_remove"])

    # if alpaca, create a test and validation set from the training set
    if name == "alpaca":
        ds = ds["train"].train_test_split(test_size=0.2, seed=42)
        temp_ds = ds.pop("test")
        temp_ds = temp_ds.train_test_split(test_size=0.5, seed=42)
        ds["test"] = temp_ds["train"]
        ds["validation"] = temp_ds["test"]

    logging.info("Loading dataset done")
    return ds


def prepare_test_dataloader(
    dataset: datasets.Dataset, tokenizer: PreTrainedTokenizerBase, seqlen: int = 2048, batch_size: int = 1
) -> DataLoader[Dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a test dataset. This dataloader should be used when comparing WikiText2 perplexities with other papers, e.g. SparseGPT (arxiv.org/abs/2301.00774).

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        seqlen: The sequence length of sequences in the dataset.
        batch_size: The batch size.

    Returns:
        A DataLoader.
    """

    logging.info(f"Preparing test dataloader")

    class TestDataset(Dataset):
        def __init__(self, ds, tokenizer, seqlen=2048):
            """Tokenize the entire dataset and reshape it into sequences of length seqlen."""

            tokenized_ds = tokenizer("\n\n".join(ds['text']), return_tensors='pt')
            nsamples = tokenized_ds.input_ids.numel() // seqlen
            input_ids = tokenized_ds.input_ids[0, : nsamples * seqlen]
            input_ids = input_ids.reshape(nsamples, seqlen)
            attn_mask = tokenized_ds.attention_mask[0, : nsamples * seqlen]
            attn_mask = attn_mask.reshape(nsamples, seqlen)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    loader = DataLoader(test_ds, batch_size=batch_size)
    logging.info(f"Preparing test dataloader done")
    return loader


def prepare_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seqlen: int = 2048,
    batch_size: int = 1,
    nsamples: int = 128,
    varied_seqlen: bool = False,
    seed=42,
) -> DataLoader[Dict[str, torch.Tensor]]:
    """
    Get a DataLoader from a dataset.

    Args:
        dataset: The dataset to create a dataloader from.
        tokenizer: The tokenizer to use.
        max_seqlen: The maximum sequence length, used for truncation of sequences in the dataset.
        batch_size: The batch size.
        nsamples: The number of samples to produce.
        varied_seqlen: If False, concatenate multiple examples from the dataset into one example until max_seqlen is reached.
        seed: The seed for sampling the dataset.

    Returns:
        A DataLoader.
    """
    logging.info(f"Preparing dataloader")

    if not varied_seqlen and not nsamples:
        logging.warning(
            "varied_seqlen=False, but nsamples is not specified. This will lead to tokenization of the entire dataset, which will be slow."
        )

    data_name = dataset.column_names[0]
    ds = dataset.filter(lambda x: len(x[data_name]) > 0)

    if not varied_seqlen:
        # create a new dataset where each example is a concatenation of multiple examples of total length = max_seqlen.
        data_list = ds[data_name]
        new_data_list = []

        torch.manual_seed(seed)
        indices = list(range(len(data_list)))

        while len(new_data_list) < nsamples and len(indices) > 0:
            start_idx = torch.randint(0, len(indices), (1,)).item()
            idx = start_idx
            tokens = []
            while len(tokens) < max_seqlen and idx < len(indices):
                item = data_list[indices[idx]]
                sep = "" if not tokens else "\n\n"
                tokens += tokenizer.tokenize(sep + item)
                idx += 1

            indices = indices[:start_idx] + indices[idx:]  # remove the used indices

            if len(tokens) >= max_seqlen:
                tokens = tokens[:max_seqlen]  # truncate to max_seqlen
                new_data_list.append(tokenizer.convert_tokens_to_string(tokens))

        ds = datasets.Dataset.from_dict({data_name: new_data_list})

    def tokenize(data_batch):
        # tokenize then pad each batch according to the longest sequence in the batch
        batch = tokenizer(
            data_batch[data_name],
            padding="longest",
            max_length=max_seqlen,
            truncation=True,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

    # tokenize lazily
    ds.set_transform(tokenize)

    torch.manual_seed(seed)
    sampler = SubsetRandomSampler(torch.randperm(len(ds))[:nsamples])

    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    logging.info(f"Preparing dataloader done")
    return loader


def get_block_importance_by_ppl(
    model,
    dataloader,
    pad_token_id=None,
    silence=True
) -> float:
    return evaluate_ppl(model, dataloader, pad_token_id, silence)


def get_block_importance_by_lm_eval_acc(
    hflm,
    task_names,
    num_calibration_samples=None,
) -> float:
    # Evaluate on selected tasks
    results = evaluator.simple_evaluate(hflm, tasks=task_names, log_samples=False, limit=num_calibration_samples)['results']

    def calculate_avg_accuracy(task_names, results):
        n_tasks = len(task_names)
        acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
        return round(acc_cumul / n_tasks, 4) * 100

    acc_avg = calculate_avg_accuracy(task_names, results)
    return -1 * acc_avg


importance_metric_func_mapping = {
    "ppl": get_block_importance_by_ppl,
    "lm_eval_acc": get_block_importance_by_lm_eval_acc
}


# only supports llama, qwen2
DEPENDENCY_GROUPS = {
    "down_proj": "up_proj",
    "o_proj": "q_proj"
}


def get_num_layers(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return model.config.num_hidden_layers
    else:
        raise ValueError


def get_hidden_size(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return model.config.hidden_size
    else:
        raise ValueError


def get_intermediate_size(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return model.config.intermediate_size
    else:
        raise ValueError


def get_num_attention_heads(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return model.config.num_attention_heads
    else:
        raise ValueError


def get_num_kv_heads(model):
    if model.config.model_type in ["llama", "qwen2", "mistral"]:
        return model.config.num_key_value_heads
    elif model.config.model_type in ["baichuan"]:
        return get_num_attention_heads(model)
    else:
        raise ValueError


def get_head_size(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return get_hidden_size(model) // get_num_attention_heads(model)
    else:
        raise ValueError


def get_attn_key(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return "self_attn"
    else:
        raise ValueError


def get_k_key(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return "k_proj"
    else:
        raise ValueError


def get_v_key(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return "v_proj"
    else:
        raise ValueError


def get_mlp_key(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return "mlp"
    else:
        raise ValueError


def get_layers(model):
    if model.config.model_type in ["llama", "qwen2", "baichuan", "mistral"]:
        return model.model.layers
    else:
        raise ValueError


def reorder_in_mlp_block(block):
    module_to_reorder = {}
    for name, module in block.named_children():
        if isinstance(module, torch.nn.Linear):
            module_to_reorder[name] = module
    # cumulative importance
    cumulative_filters_importance = None
    for name, module in module_to_reorder.items():
        if name in DEPENDENCY_GROUPS:
            continue
        weight_tensor = module.weight
        weight_tensor = weight_tensor.transpose(0, 0).contiguous()
        filters_importance = torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=1, dim=1)
        if cumulative_filters_importance is None:
            cumulative_filters_importance = filters_importance
        else:
            cumulative_filters_importance += filters_importance

    _, reorder_indexes = torch.sort(cumulative_filters_importance, dim=0, descending=True)
    for name, module in module_to_reorder.items():
        if name in DEPENDENCY_GROUPS:
            module.weight.data = torch.index_select(module.weight.data, 1, reorder_indexes)
        else:
            module.weight.data = torch.index_select(module.weight.data, 0, reorder_indexes)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, reorder_indexes)


def reorder_in_attn_block(block, model):
    module_to_reorder = {}
    for name, module in block.named_children():
        if isinstance(module, torch.nn.Linear):
            module_to_reorder[name] = module
    
    num_key_value_heads = get_num_kv_heads(model)
    num_attention_heads = get_num_attention_heads(model)
    num_key_value_groups = num_attention_heads // num_key_value_heads
    head_size = get_head_size(model)
    hidden_size = get_hidden_size(model)
    k_proj_key, v_proj_key = get_k_key(model), get_v_key(model)
    n_reps_head_size = head_size * num_key_value_groups
    
    # cumulative importance
    cumulative_filters_importance = None
    for name, module in module_to_reorder.items():
        if name in DEPENDENCY_GROUPS:
            continue
        weight_tensor = module.weight
        if name in [k_proj_key, v_proj_key]:
            weight_tensor = weight_tensor.view(-1, head_size, hidden_size)
            weight_tensor = weight_tensor[:, None, :].expand(num_key_value_heads, num_key_value_groups, head_size, hidden_size)
            weight_tensor = weight_tensor.reshape(num_key_value_heads * num_key_value_groups, head_size, hidden_size)
            weight_tensor = weight_tensor.view(hidden_size, hidden_size)
        weight_tensor = weight_tensor.transpose(0, 0).contiguous()
        filters_importance = torch.norm(weight_tensor.view(weight_tensor.shape[0], -1), p=1, dim=1)
        if cumulative_filters_importance is None:
            cumulative_filters_importance = filters_importance
        else:
            cumulative_filters_importance += filters_importance

    importance_per_head = torch.mean(cumulative_filters_importance.view(-1, n_reps_head_size), dim=-1)
    _, reorder_indexes_per_head = torch.sort(importance_per_head, dim=0, descending=True)
    reorder_indexes = (
        torch.arange(0, n_reps_head_size)
        .int()
        .repeat(len(importance_per_head))
        .to(reorder_indexes_per_head.device)
        + torch.repeat_interleave(reorder_indexes_per_head, n_reps_head_size) * n_reps_head_size
    )
    reorder_indexes_for_kv = (
        torch.arange(0, head_size)
        .int()
        .repeat(len(importance_per_head))
        .to(reorder_indexes_per_head.device)
        + torch.repeat_interleave(reorder_indexes_per_head, head_size) * head_size
    )

    for name, module in module_to_reorder.items():
        if name in DEPENDENCY_GROUPS:
            module.weight.data = torch.index_select(module.weight.data, 1, reorder_indexes)
        else:
            cur_reorder_indexes = reorder_indexes
            if name in [k_proj_key, v_proj_key]:
                cur_reorder_indexes = reorder_indexes_for_kv
            module.weight.data = torch.index_select(module.weight.data, 0, cur_reorder_indexes)
            if module.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, cur_reorder_indexes)
