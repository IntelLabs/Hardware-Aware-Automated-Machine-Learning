import argparse
import csv
import json
import logging
import os
import random

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import utils

DEPENDENCY_GROUPS = {
    "down_proj": "up_proj"
}
TASKS = ["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]


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
        return torch.nn.functional.linear(x, masked_weight, masked_bias)

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


def replace_with_mask_linear(model):
    replaced_module = {}
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            mask_linear = MaskLinear(module)
            setattr(model, name, mask_linear)
            replaced_module[name] = mask_linear
    return replaced_module


def mask_self_attn(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mamba") and hasattr(layers[idx].mamba, "mask_self_attn")
        if idx in layer_indices:
            layers[idx].mamba.mask_self_attn = True
        else:
            layers[idx].mamba.mask_self_attn = False


def mask_ssm(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mamba") and hasattr(layers[idx].mamba, "mask_ssm")
        if idx in layer_indices:
            layers[idx].mamba.mask_ssm = True
        else:
            layers[idx].mamba.mask_ssm = False

def mask_hymba_block(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_hymba_block")
        if idx in layer_indices:
            layers[idx].mask_hymba_block = True
        else:
            layers[idx].mask_hymba_block = False


def mask_moe(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_moe")
        if idx in layer_indices:
            layers[idx].mask_moe = True
        else:
            layers[idx].mask_moe = False


def mask_width(groups, wdith_config):
    for group_idx, value in wdith_config.items():
        group = groups[int(group_idx)]
        for name, module in group.items():
            if name in DEPENDENCY_GROUPS:
                module.set_in_mask_size(value)
            else:
                module.set_out_mask_size(value)


def prune(model, pruning_config, groups_for_width_pruning=None):
    if "pruned_self_attn_idx" in pruning_config:
        mask_self_attn(model.model.layers, pruning_config["pruned_self_attn_idx"])
    if "pruned_ssm_idx" in pruning_config:
        mask_ssm(model.model.layers, pruning_config["pruned_ssm_idx"])
    if "pruned_hymba_block_idx" in pruning_config:
        mask_hymba_block(model.model.layers, pruning_config["pruned_hymba_block_idx"])
    if "pruned_moe_idx" in pruning_config:
        mask_moe(model.model.layers, pruning_config["pruned_moe_idx"])
    
    # width pruning
    if "pruned_mlp_channels" in pruning_config and groups_for_width_pruning:
        mask_width(groups_for_width_pruning, pruning_config["pruned_mlp_channels"])


# The SyncLogger class is used to record the search process
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
        help="Path to Hymba model."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prune_result",
        help="Directory to save the pruning and evaluation results."
    )
    parser.add_argument(
        "--do_prune",
        action="store_true",
        help="Flag to indicate whether to perform pruning."
    )
    parser.add_argument(
        "--block_pruning_targets",
        type=str,
        default="self_attn",
    )
    parser.add_argument(
        "--num_block_pruning_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_width_pruning_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mlp_channel_group_size",
        type=int,
        default=1024,
        help="Number of channels for each group (MLP)."
    )
    parser.add_argument(
        "--weight_reorder",
        action="store_true",
        help="Flag to indicate whether to perform weight reorder in MLP."
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
        "--num_calibration_samples",
        type=int,
        default=256,
        help="Number of samples to use for calibration during pruning."
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
        help="Path to the pruned model configuration file."
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
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # pruning config
    do_prune = args.do_prune
    block_pruning_targets = args.block_pruning_targets
    num_block_pruning_steps = args.num_block_pruning_steps
    num_width_pruning_steps = args.num_width_pruning_steps
    mlp_channel_group_size = args.mlp_channel_group_size
    weight_reorder = args.weight_reorder
    importance_metric = args.importance_metric
    calibration_dataset = args.calibration_dataset
    num_calibration_samples = args.num_calibration_samples
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

    logging.info(f"Loading model {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of total parameters: {total_params}")
    logger.data["total_params"] = total_params
    num_layers = model.config.num_hidden_layers
    block_pruning_targets = block_pruning_targets.split(",")

    if weight_reorder:
        logger.data["weight_reorder"] = True
        for layer in model.model.layers:
            reorder_in_mlp_block(layer.moe.experts[0])
    
    if do_prune:
        
        # calibration dataset
        dataset = utils.get_dataset(calibration_dataset)
        test_dataset = dataset["test"]
        test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples))
        calibration_dataloader = utils.prepare_test_dataloader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            seqlen=1024,
            batch_size=4
        )
        importance_metric_func = utils.importance_metric_func_mapping[importance_metric]

        logging.info(f"Number of block pruning steps : {num_block_pruning_steps} (targets: {block_pruning_targets})")
        logging.info(f"Number of width pruning steps : {num_width_pruning_steps}")
        step = 0
        logger.data["pruning_results"] = []
        num_evals = 0
        pruning_config = {
            "pruned_self_attn_idx": [],
            "pruned_ssm_idx": [],
            "pruned_hymba_block_idx": [],
            "pruned_moe_idx": [],
            "pruned_mlp_channels": {i: model.config.intermediate_size for i in range(num_layers)},
        }
        layer_ids = list(range(num_layers))

        # hymba block pruning limitation (KV sharing)
        hymba_block_candidate_ids = None
        if "hymba_block" in block_pruning_targets:
            hymba_block_candidate_ids = [group[-1] for group in model.config.kv_reuse_group] + model.config.global_attn_idx
            hymba_block_candidate_ids.sort()
            logging.info(f"Due to the KV Sharing mechanism, the candidate pruning targets are restricted to the hymba blocks in layers {hymba_block_candidate_ids}.")

        logging.info(f"Start Block pruning (target: {block_pruning_targets}) ...")
        for step in range(num_block_pruning_steps):
            best_candidate = None
            lowest_importance = float('inf')

            # candidates
            candidates = []
            for pruning_target in block_pruning_targets:
                candidate_layer_ids = hymba_block_candidate_ids if pruning_target == "hymba_block" else layer_ids
                for layer_id in candidate_layer_ids:
                    if layer_id not in pruning_config[f"pruned_{pruning_target}_idx"]:
                        if pruning_target in ["self_attn", "ssm"] and layer_id in pruning_config[f"pruned_hymba_block_idx"]:
                            continue
                        candidates.append([pruning_target, layer_id])
            logging.info(f"Candidates in step {step}: {candidates}")
            for candidate in tqdm(candidates):
                target, target_id = candidate
                num_evals += 1
                pruning_config[f"pruned_{target}_idx"].append(target_id)
                prune(model, pruning_config)

                # the importance of the current pruned model
                target_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader,
                    pad_token_id=model.config.pad_token_id
                )
                # revert
                pruning_config[f"pruned_{target}_idx"].pop()
                prune(model, pruning_config)

                # expect to prune the least important one
                if target_importance < lowest_importance:
                    lowest_importance = target_importance
                    best_candidate = candidate

            # pruning
            cur_pruning_target = best_candidate[0]
            pruning_config[f"pruned_{cur_pruning_target}_idx"].append(best_candidate[1])
            prune(model, pruning_config)
            config_path = os.path.join(pruning_config_path, f"config.{args.block_pruning_targets}.{step}.json")
            with open(config_path, "w") as f:
                json.dump(pruning_config, f, indent=4)

            info = {
                "step": step,
                "pruning_target": cur_pruning_target,
                "importance": lowest_importance,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} - Number of Candidates: {len(candidates)}")
            logging.info(f"Step {step} - Pruning Target: {cur_pruning_target}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Importance: {lowest_importance}")

        logging.info(f"Completed Block pruning (target: {block_pruning_targets}).")

        pruned_mlp_channels = pruning_config["pruned_mlp_channels"]
        logging.info(f"Start MLP width (in moe expert, ffn) pruning...")

        mlp_groups = []
        for layer in model.model.layers:
            mlp_groups.append(replace_with_mask_linear(layer.moe.experts[0]))

        for step in range(num_block_pruning_steps, num_block_pruning_steps + num_width_pruning_steps):
            best_candidate = None
            lowest_channel_groups_importance = float('inf')
            candidates = []
            for mlp_id, width in pruned_mlp_channels.items():
                if (width - mlp_channel_group_size) > 0 and (mlp_id not in pruning_config["pruned_moe_idx"]):
                    candidate = (mlp_id, width - mlp_channel_group_size)
                    candidates.append(candidate)

            for candidate in tqdm(candidates):
                num_evals += 1

                mlp_id, pruned_width = candidate
                pre_pruned_width = pruned_mlp_channels[mlp_id]
                pruned_mlp_channels[mlp_id] = pruned_width
                prune(model, pruning_config, groups_for_width_pruning=mlp_groups)

                # the importance of the current channel-groups
                channel_groups_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader,
                    pad_token_id=model.config.pad_token_id
                )
                pruned_mlp_channels[mlp_id] = pre_pruned_width
                prune(model, pruning_config, groups_for_width_pruning=mlp_groups)

                # expect to prune the least important channel-groups
                if channel_groups_importance < lowest_channel_groups_importance:
                    lowest_channel_groups_importance = channel_groups_importance
                    best_candidate = candidate

            # pruning
            pruned_mlp_channels[best_candidate[0]] = best_candidate[1]
            prune(model, pruning_config, groups_for_width_pruning=mlp_groups)

            config_path = os.path.join(pruning_config_path, f"config.mlp_width.{step}.json")
            pruning_config_str = {str(dim): value for dim, value in pruning_config.items()}
            with open(config_path, "w") as f:
                json.dump(pruning_config_str, f, indent=4)

            info = {
                "step": step,
                "pruning_target": "mlp_width",
                "importance": lowest_channel_groups_importance,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs MLP width pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidates)}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Importance: {lowest_channel_groups_importance}")
        
        logging.info(f"Completed MLP width (in moe expert, ffn) pruning.")

        logger.data["num_evals"] = num_evals
        logger.update()

        # save the last pruning config
        config_path = os.path.join(output_path, f"pruning_config.json")
        with open(config_path, "w") as f:
            json.dump(pruning_config, f, indent=4)

    if do_eval:
        logging.info(f"Start evaluation...")

        if pruned_model_config_file is None:
            assert do_prune
            pruned_model_config_file = config_path

        # Load pruning results
        with open(pruned_model_config_file, "r") as f:
            pruned_config = json.load(f)

        # prune
        mlp_groups = []
        for layer in model.model.layers:
            mlp_groups.append(replace_with_mask_linear(layer.moe.experts[0]))
        prune(model, pruned_config, groups_for_width_pruning=mlp_groups)
        logging.info(f"Detect a pruned model config: {pruned_config}")

        # Evaluate on selected tasks
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        logging.info(f"Selected Tasks: {TASKS}")

        results = evaluator.simple_evaluate(lm, tasks=TASKS, num_fewshot=0, batch_size=batch_size, log_samples=False)['results']

        metric_vals = {}
        total_score = 0.0
        for task, result in results.items():
            res = result['acc,none']
            total_score += res
            metric_vals[task] = round(res, 3) * 100
        logging.info(json.dumps(metric_vals, indent=4))

        n_tasks = len(TASKS)
        acc_avg = round(total_score / n_tasks, 3) * 100
        logging.info(f"Average accuracy across tasks: {acc_avg}")

        # Save evaluation results
        overall_results = {
            "total_params": total_params,
            "5cs_acc_avg": acc_avg,
            **metric_vals
        }
        result_file_name = ".".join(pruned_model_config_file.split("/")[-1].split(".")[:-1])
        eval_result_path = os.path.join(output_path, f"eval.res.{result_file_name}.json")
        with open(eval_result_path, "w") as f:
            json.dump(overall_results, f, indent=4)

        eval_result_csv_path = os.path.join(output_path, f"eval.res.{result_file_name}.csv")
        columns = ["total_params"] + TASKS + ["5cs_acc_avg"]
        with open(eval_result_csv_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow(overall_results)

        logger.data[f"evaluation_{result_file_name}"] = overall_results
        logger.update()

if __name__ == "__main__":
    main()
