import argparse
import csv
import json
import logging
import os
import random
import torch
import re

import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import utils


class MaskLinear(torch.nn.Module):
    def __init__(self, original_linear: torch.nn.Linear):
        super(MaskLinear, self).__init__()
        self.original_linear = original_linear
        self.out_mask_size = original_linear.out_features
        self.in_mask_size = original_linear.in_features
        self.out_mask = torch.ones(self.original_linear.out_features, dtype=torch.bool)
        self.in_mask = torch.ones(self.original_linear.in_features, dtype=torch.bool)

    def forward(self, x):
        # Apply the mask to the weight and bias
        masked_weight = self.original_linear.weight[self.out_mask, :][:, self.in_mask]
        masked_bias = self.original_linear.bias[self.out_mask] if self.original_linear.bias is not None else None
        return F.linear(x, masked_weight, masked_bias)

    def set_out_mask_size(self, new_out_mask_size: int):
        if new_out_mask_size > self.original_linear.out_features or new_out_mask_size <= 0:
            raise ValueError("Invalid output mask size")
        self.out_mask_size = new_out_mask_size
        self.out_mask = torch.ones(self.original_linear.out_features, dtype=torch.bool)
        self.out_mask[new_out_mask_size:] = False

    def set_in_mask_size(self, new_in_mask_size: int):
        if new_in_mask_size > self.original_linear.in_features or new_in_mask_size <= 0:
            raise ValueError("Invalid input mask size")
        self.in_mask_size = new_in_mask_size
        self.in_mask = torch.ones(self.original_linear.in_features, dtype=torch.bool)
        self.in_mask[new_in_mask_size:] = False



def count_parameters(model):
    def count_masked_params(module):
        total = 0
        for name, sub_module in module.named_children():
            if isinstance(sub_module, MaskLinear):
                masked_weight = sub_module.original_linear.weight[sub_module.out_mask, :][:, sub_module.in_mask]
                total += masked_weight.numel()
                if sub_module.original_linear.bias is not None:
                    masked_bias = sub_module.original_linear.bias[sub_module.out_mask]
                    total += masked_bias.numel()
            else:
                total += sum(p.numel() for p in sub_module.parameters())
        return total

    # Get the parameters of layers
    layers = utils.get_layers(model)
    layers_params = {p for layer in layers for p in layer.parameters()}

    # Calculate the total number of parameters excluding layers (e.g., embedding layer)
    total_params = sum(p.numel() for p in model.parameters() if p not in layers_params)

    for layer in layers:
        attn_module = getattr(layer, utils.get_attn_key(model))
        mlp_module = getattr(layer, utils.get_mlp_key(model))

        # Count parameters in the layer excluding `attn_module` and `mlp_module`
        attn_module_params = {p for p in attn_module.parameters()}
        mlp_module_params = {p for p in mlp_module.parameters()}
        total_params += sum(
            p.numel() for p in layer.parameters() if p not in attn_module_params and p not in mlp_module_params)

        # Count parameters in `attn_module`
        if not getattr(layer, "mask_attn", False):
            total_params += count_masked_params(attn_module)

        # Count parameters in `mlp_module`
        if not getattr(layer, "mask_mlp", False):
            total_params += count_masked_params(mlp_module)

    return total_params


def replace_with_mask_linear(model):
    replaced_module = {}
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            mask_linear = MaskLinear(module)
            setattr(model, name, mask_linear)
            replaced_module[name] = mask_linear
    return replaced_module


def mask_mlp(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_mlp")
        if idx in layer_indices:
            layers[idx].mask_mlp = True
        else:
            layers[idx].mask_mlp = False


def mask_attn(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_attn")
        if idx in layer_indices:
            layers[idx].mask_attn = True
        else:
            layers[idx].mask_attn = False


def mask_mlp_width(groups, wdith_config):
    for group_idx, value in wdith_config.items():
        group = groups[int(group_idx)]
        for name, module in group.items():
            # down_proj (final linear projection in mlp block)
            if name in utils.DEPENDENCY_GROUPS:
                module.set_in_mask_size(value)
            else:
                module.set_out_mask_size(value)


def mask_attn_head(groups, wdith_config, model):
    num_key_value_heads = utils.get_num_kv_heads(model)
    num_attention_heads = utils.get_num_attention_heads(model)
    num_key_value_groups = num_attention_heads // num_key_value_heads
    k_proj_key, v_proj_key = utils.get_k_key(model), utils.get_v_key(model)
    for group_idx, value in wdith_config.items():
        group = groups[int(group_idx)]
        for name, module in group.items():
            # out_proj (final linear projection in attention block)
            if name in utils.DEPENDENCY_GROUPS:
                module.set_in_mask_size(value)
            else:
                if name in [k_proj_key, v_proj_key]:
                    module.set_out_mask_size(value // num_key_value_groups)
                else:
                    module.set_out_mask_size(value)


def prune(model, pruning_config, groups_for_mlp_width_pruning=None, groups_for_attn_width_pruning=None):
    # Attn/MLP Block pruning
    if "pruned_attn_idx" in pruning_config:
        mask_attn(utils.get_layers(model), pruning_config["pruned_attn_idx"])
    if "pruned_mlp_idx" in pruning_config:
        mask_mlp(utils.get_layers(model), pruning_config["pruned_mlp_idx"])

    # MLP width pruning
    if "pruned_mlp_width" in pruning_config and groups_for_mlp_width_pruning:
        mask_mlp_width(groups_for_mlp_width_pruning, pruning_config["pruned_mlp_width"])
    # Attn head pruning
    if "pruned_attn_width" in pruning_config and groups_for_attn_width_pruning:
        mask_attn_head(groups_for_attn_width_pruning, pruning_config["pruned_attn_width"], model)


# The SyncLogger class is used to record the pruning process
class SyncLogger:
    def __init__(self, filename):
        # During initialization, try to load the dictionary from the specified file
        self.filename = filename
        self.data = {}
        self._load()

    def _load(self):
        # Load the dictionary from the file if it exists
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.data = json.load(f)

    def update(self):
        # Update the dictionary and write the latest dictionary into the file
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prune_result",
        help="Directory to save the pruning and evaluation results."
    )
    parser.add_argument(
        "--weight_reorder",
        action="store_true",
        help="Flag to indicate whether to perform weight reorder."
    )
    parser.add_argument(
        "--do_prune",
        action="store_true",
        help="Flag to indicate whether to perform pruning."
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=20.0,
        help="Target pruning ratio."
    )
    parser.add_argument(
        "--pruning_distribution",
        type=str,
        default="44:52:4",
        help="Pruning ratio distribution for three granularities."
    )
    parser.add_argument(
        "--mlp_channel_group_size",
        type=int,
        default=1024,
        help="Number of channels for each group (MLP)."
    )
    parser.add_argument(
        "--attn_channel_group_size",
        type=int,
        default=128,
        help="Number of channels for each group (Attn), generally a multiple of the head dimension."
    )
    parser.add_argument(
        "--importance_metric",
        type=str,
        default="ppl",
        choices=["ppl"],
        help="Metric for calculating importance."
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="alpaca",
        choices=["alpaca", "c4", "ptb", "wikitext2"]
    )
    parser.add_argument(
        "--num_calibration_samples_block",
        type=int,
        default=256,
        help="Number of samples to use for calibration during depth (block) pruning."
    )
    parser.add_argument(
        "--num_calibration_samples_width",
        type=int,
        default=128,
        help="Number of samples to use for calibration during width pruning."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Flag to indicate whether to perform evaluation."
    )
    parser.add_argument(
        "--pruned_model_config_file",
        type=str,
        default=None,
        help="Path to the pruned model configuration file (for evaluation)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation."
    )

    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path
    weight_reorder = args.weight_reorder
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # pruning config
    do_prune = args.do_prune
    mlp_channel_group_size = args.mlp_channel_group_size
    attn_channel_group_size = args.attn_channel_group_size
    target_ratio = args.target_ratio
    pruning_distribution = args.pruning_distribution
    importance_metric = args.importance_metric
    calibration_dataset = args.calibration_dataset
    num_calibration_samples_block = args.num_calibration_samples_block
    num_calibration_samples_width = args.num_calibration_samples_width
    pruning_config_path = os.path.join(output_path, "pruned_model_configs")
    os.makedirs(pruning_config_path, exist_ok=True)

    # eval
    do_eval = args.do_eval
    pruned_model_config_file = args.pruned_model_config_file
    batch_size = args.batch_size

    set_seed(42)
    log_file_path = os.path.join(output_path, "log.json")
    logger = SyncLogger(log_file_path)

    # save args
    if do_prune:
        logger.data["args"] = vars(args)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="float16",
    ).eval()
    total_params = count_parameters(model)
    assert total_params == sum(p.numel() for p in model.parameters())

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.data["total_params"] = total_params
    attn_ids = list(range(utils.get_num_layers(model)))
    mlp_ids = list(range(utils.get_num_layers(model)))
    logger.data["weight_reorder"] = False

    if weight_reorder:
        logger.data["weight_reorder"] = True
        for layer in utils.get_layers(model):
            utils.reorder_in_attn_block(getattr(layer, utils.get_attn_key(model)), model=model)
            utils.reorder_in_mlp_block(getattr(layer, utils.get_mlp_key(model)))

    if do_prune:
        # check `attn_channel_group_size`
        head_size = utils.get_head_size(model)
        assert attn_channel_group_size % head_size == 0
        num_attention_heads = utils.get_num_attention_heads(model)
        num_kv_heads = utils.get_num_kv_heads(model)
        if num_attention_heads != num_kv_heads:
            num_key_value_groups = num_attention_heads // num_kv_heads
            assert attn_channel_group_size % num_key_value_groups == 0 and (attn_channel_group_size // num_key_value_groups) % head_size == 0

        logging.info(f"Start pruning...")
        # calibration dataset
        dataset = utils.get_dataset(calibration_dataset)
        test_dataset = dataset["test"]
        test_dataset_width = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples_width))
        calibration_dataloader_width = utils.prepare_test_dataloader(
            dataset=test_dataset_width,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        test_dataset_block = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples_block))
        calibration_dataloader_block = utils.prepare_test_dataloader(
            dataset=test_dataset_block,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        importance_metric_func = utils.importance_metric_func_mapping[importance_metric]

        pruning_distribution = list(map(float, pruning_distribution.split(":")))
        pruning_distribution = [ratio / sum(pruning_distribution) for ratio in pruning_distribution]
        block_target_ratio = target_ratio * pruning_distribution[0]
        mlp_width_target_ratio = block_target_ratio + target_ratio * pruning_distribution[1]
        attn_head_target_ratio = target_ratio

        logging.info(f"Target ratio of different dimensions: {block_target_ratio}, {mlp_width_target_ratio}, {attn_head_target_ratio}")
        current_ratio = 0.0
        step = 0
        logger.data["pruning_results"] = []
        num_evals = 0
        pruning_config = {
            "pruned_attn_idx": [],
            "pruned_mlp_idx": [],
            "pruned_attn_width": {i: utils.get_hidden_size(model) for i in range(utils.get_num_layers(model))},
            "pruned_mlp_width": {i: utils.get_intermediate_size(model) for i in range(utils.get_num_layers(model))}
        }

        logging.info(f"Start Attn/MLP block pruning...")
        while current_ratio < block_target_ratio:
            best_candidate = None
            lowest_importance = float('inf')

            # Attn
            pruned_target_idx = pruning_config["pruned_attn_idx"]
            candidate_attn_blocks = [target_id for target_id in attn_ids if target_id not in pruned_target_idx]
            for target_id in tqdm(candidate_attn_blocks, desc="Attn"):
                num_evals += 1
                pruned_target_idx.append(target_id)
                prune(model, pruning_config)
                target_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader_block,
                    pad_token_id=tokenizer.eos_token_id
                )
                # revert
                pruned_target_idx.pop()
                prune(model, pruning_config)
                # expect to prune the least important block
                if target_importance < lowest_importance:
                    lowest_importance = target_importance
                    best_candidate = ("pruned_attn_idx", target_id)

            # MLP
            pruned_target_idx = pruning_config["pruned_mlp_idx"]
            candidate_mlp_blocks = [target_id for target_id in mlp_ids if target_id not in pruned_target_idx]
            for target_id in tqdm(candidate_mlp_blocks, desc="MLP"):
                num_evals += 1
                pruned_target_idx.append(target_id)
                prune(model, pruning_config)
                target_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader_block,
                    pad_token_id=tokenizer.eos_token_id
                )
                # revert
                pruned_target_idx.pop()
                prune(model, pruning_config)
                # expect to prune the least important block
                if target_importance < lowest_importance:
                    lowest_importance = target_importance
                    best_candidate = ("pruned_mlp_idx", target_id)

            # pruning
            pruning_config[best_candidate[0]].append(best_candidate[1])
            prune(model, pruning_config)
            pruned_model_params = count_parameters(model)
            current_ratio = (1 - pruned_model_params / total_params) * 100

            config_path = os.path.join(pruning_config_path, f"config.block.{step}.json")
            with open(config_path, "w") as f:
                json.dump(pruning_config, f, indent=4)

            pruning_target = re.search(r'pruned_(.*)_idx', best_candidate[0]).group(1)
            info = {
                "step": step,
                "pruning_target": pruning_target,
                "importance": lowest_importance,
                "ratio": current_ratio,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs Attn/MLP block pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidate_mlp_blocks) + len(candidate_attn_blocks)}")
            logging.info(f"Step {step} - Pruning Target: {pruning_target}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Pruning Ratio: {current_ratio}")
            logging.info(f"Step {step} - Importance: {lowest_importance}")
            step += 1

        logging.info(f"Complete Attn/MLP block pruning.")
    
        pruned_mlp_width = pruning_config["pruned_mlp_width"]
        logging.info(f"Start MLP width pruning...")

        mlp_groups = []
        for layer in utils.get_layers(model):
            mlp_groups.append(replace_with_mask_linear(getattr(layer, utils.get_mlp_key(model))))

        while current_ratio < mlp_width_target_ratio:
            best_candidate = None
            lowest_channel_groups_importance = float('inf')
            candidates = []
            for mlp_id, width in pruned_mlp_width.items():
                if (width - mlp_channel_group_size) > 0 and (mlp_id not in pruning_config["pruned_mlp_idx"]):
                    candidate = (mlp_id, width - mlp_channel_group_size)
                    candidates.append(candidate)

            for candidate in tqdm(candidates):
                mlp_id, pruned_width = candidate
                pre_pruned_width = pruned_mlp_width[mlp_id]
                pruned_mlp_width[mlp_id] = pruned_width
                prune(model, pruning_config, groups_for_mlp_width_pruning=mlp_groups)

                # the importance of the current channel-groups
                channel_groups_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader_width,
                    pad_token_id=tokenizer.eos_token_id
                )
                pruned_mlp_width[mlp_id] = pre_pruned_width
                prune(model, pruning_config, groups_for_mlp_width_pruning=mlp_groups)

                # expect to prune the least important channel-groups
                if channel_groups_importance < lowest_channel_groups_importance:
                    lowest_channel_groups_importance = channel_groups_importance
                    best_candidate = candidate

            # pruning
            pruned_mlp_width[best_candidate[0]] = best_candidate[1]
            prune(model, pruning_config, groups_for_mlp_width_pruning=mlp_groups)

            pruned_model_params = count_parameters(model)
            current_ratio = (1 - pruned_model_params / total_params) * 100

            config_path = os.path.join(pruning_config_path, f"config.mlp_width.{step}.json")
            pruning_config_str = {str(dim): value for dim, value in pruning_config.items()}
            with open(config_path, "w") as f:
                json.dump(pruning_config_str, f, indent=4)

            info = {
                "step": step,
                "pruning_target": "MLP width",
                "importance": lowest_channel_groups_importance,
                "ratio": current_ratio,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs MLP width pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidates)}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Pruning Ratio: {current_ratio}")
            logging.info(f"Step {step} - Importance: {lowest_channel_groups_importance}")
            step += 1

        logging.info(f"Complete MLP width pruning.")

        pruned_attn_width = pruning_config["pruned_attn_width"]
        logging.info(f"Start Attn head pruning...")

        attn_groups = []
        for layer in utils.get_layers(model):
            attn_groups.append(replace_with_mask_linear(getattr(layer, utils.get_attn_key(model))))

        while current_ratio < attn_head_target_ratio:
            best_candidate = None
            lowest_head_importance = float('inf')
            candidates = []
            for attn_id, width in pruned_attn_width.items():
                if (width - attn_channel_group_size) > 0 and (attn_id not in pruning_config["pruned_attn_idx"]):
                    candidate = (attn_id, width - attn_channel_group_size)
                    candidates.append(candidate)

            for candidate in tqdm(candidates):
                attn_id, pruned_width = candidate
                pre_pruned_width = pruned_attn_width[attn_id]
                pruned_attn_width[attn_id] = pruned_width
                prune(model, pruning_config, groups_for_attn_width_pruning=attn_groups)

                # the importance of the current channel-groups
                head_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader_width,
                    pad_token_id=tokenizer.eos_token_id
                )
                pruned_attn_width[attn_id] = pre_pruned_width
                prune(model, pruning_config, groups_for_attn_width_pruning=attn_groups)

                # expect to prune the least important channel-groups
                if head_importance < lowest_head_importance:
                    lowest_head_importance = head_importance
                    best_candidate = candidate

            # pruning
            pruned_attn_width[best_candidate[0]] = best_candidate[1]
            prune(model, pruning_config, groups_for_attn_width_pruning=attn_groups)

            pruned_model_params = count_parameters(model)
            current_ratio = (1 - pruned_model_params / total_params) * 100

            config_path = os.path.join(pruning_config_path, f"config.attn_head.{step}.json")
            pruning_config_str = {str(dim): value for dim, value in pruning_config.items()}
            with open(config_path, "w") as f:
                json.dump(pruning_config_str, f, indent=4)

            info = {
                "step": step,
                "pruning_target": "Attn head",
                "importance": lowest_head_importance,
                "ratio": current_ratio,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs Attn head pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidates)}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Pruning Ratio: {current_ratio}")
            logging.info(f"Step {step} - Importance: {lowest_head_importance}")
            step += 1

        logging.info(f"Complete Attn head pruning.")
        
        logger.data["num_evals"] = num_evals
        logger.update()

        # save the last pruning config
        config_path = os.path.join(output_path, f"pruning_config.json")
        with open(config_path, "w") as f:
            json.dump(pruning_config, f, indent=4)

    if do_eval:
        logging.info(f"Start evaluation...")

        if pruned_model_config_file is None:
            assert do_prune, "`--do_prune` is not enabled. Please provide a pruned model configuration file for evaluation."
            pruned_model_config_file = os.path.join(output_path, f"pruning_config.json")

        # Load pruning results
        with open(pruned_model_config_file, "r") as f:
            pruned_config = json.load(f)

        mlp_groups = []
        for layer in utils.get_layers(model):
            mlp_groups.append(replace_with_mask_linear(getattr(layer, utils.get_mlp_key(model))))
        attn_groups = []
        for layer in utils.get_layers(model):
            attn_groups.append(replace_with_mask_linear(getattr(layer, utils.get_attn_key(model))))

        prune(model, pruned_config, groups_for_mlp_width_pruning=mlp_groups, groups_for_attn_width_pruning=attn_groups)
        pruned_model_params = count_parameters(model)
        logging.info(f"Detect a pruned model config for evaluation: {pruned_config}")

        # Evaluate on wikitext2 dataset
        dataset = utils.get_dataset("wikitext2")
        test_dataset = dataset["test"]
        test_loader = utils.prepare_test_dataloader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=1
        )
        dataset_ppl = utils.evaluate_ppl(
            model=model,
            dataloader=test_loader,
            pad_token_id=model.config.eos_token_id,
        )
        dataset_ppl = round(dataset_ppl, 2)
        logging.info(f'wikitext2 PPL: {dataset_ppl}')

        # Evaluate on selected tasks
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

        task_names = ["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"]
        logging.info(f"Selected Tasks: {task_names}")

        results = evaluator.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=batch_size, log_samples=False)['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) * 100 for task, result in
                       results.items()}
        logging.info(json.dumps(metric_vals, indent=4))

        def calculate_avg_accuracy(task_names, results):
            n_tasks = len(task_names)
            acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
            return round(acc_cumul / n_tasks, 4) * 100

        acc_avg = calculate_avg_accuracy(task_names, results)
        logging.info(f"Average accuracy across tasks: {acc_avg}")

        # Save evaluation results
        overall_results = {
            "total_params": total_params,
            "pruned_model_params": pruned_model_params,
            "ppl_wikitext2": dataset_ppl,
            "5cs_acc_avg": acc_avg,
            **metric_vals
        }
        result_file_name = ".".join(pruned_model_config_file.split("/")[-1].split(".")[:-1])
        eval_result_path = os.path.join(output_path, f"eval.res.{result_file_name}.json")
        with open(eval_result_path, "w") as f:
            json.dump(overall_results, f, indent=4)

        eval_result_csv_path = os.path.join(output_path, f"eval.res.{result_file_name}.csv")
        columns = ["total_params", "pruned_model_params", "ratio", "ppl_wikitext2"] + task_names + ["5cs_acc_avg"]
        with open(eval_result_csv_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow(overall_results)

        logger.data[f"evaluation_{result_file_name}"] = overall_results
        logger.update()

if __name__ == "__main__":
    main()
