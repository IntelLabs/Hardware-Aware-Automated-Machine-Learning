import argparse
import csv
import json
import logging
import os
import random
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import utils

DEPENDENCY_GROUPS = {
    "linear_fc2": "linear_fc1_up"
}
TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]


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


def replace_with_mask_linear(model):
    replaced_module = {}
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            mask_linear = MaskLinear(module)
            setattr(model, name, mask_linear)
            replaced_module[name] = mask_linear
    return replaced_module


def mask_mamba_block(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_mamba_block")
        if idx in layer_indices:
            layers[idx].mask_mamba_block = True
        else:
            layers[idx].mask_mamba_block = False


def mask_mlp(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_feed_forward")
        if idx in layer_indices:
            layers[idx].mask_feed_forward = True
        else:
            layers[idx].mask_feed_forward = False


def mask_mha(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_self_attn")
        if idx in layer_indices:
            layers[idx].mask_self_attn = True
        else:
            layers[idx].mask_self_attn = False


def mask_ssm(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mamba") and hasattr(layers[idx].mamba, "mask_ssm")
        if idx in layer_indices:
            layers[idx].mamba.mask_ssm = True
        else:
            layers[idx].mamba.mask_ssm = False


def mask_width(groups, wdith_config):
    for group_idx, value in wdith_config.items():
        group = groups[group_idx]
        for name, module in group.items():
            if name in DEPENDENCY_GROUPS:
                module.set_in_mask_size(value)
            else:
                module.set_out_mask_size(value)


def prune(model, pruning_config, groups_for_width_pruning=None):
    # depth pruning
    if "pruned_mamba_block_idx" in pruning_config:
        mask_mamba_block(model.model.mamba_layers, pruning_config["pruned_mamba_block_idx"])
    if "pruned_mha_idx" in pruning_config:
        mask_mha(model.model.blocks, pruning_config["pruned_mha_idx"])
    if "pruned_mlp_idx" in pruning_config:
        mask_mlp(model.model.blocks, pruning_config["pruned_mlp_idx"])
    if "pruned_ssm_idx" in pruning_config:
        mask_ssm(model.model.mamba_layers, pruning_config["pruned_ssm_idx"])

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
        help="Path to the Zamba model."
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
        "--target_block_pruning_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--target_width_pruning_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--target_ssm_pruning_steps",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--mlp_channel_group_size",
        type=int,
        default=1024,
        help="Number of channels for each group (MLP)."
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
        help="Number of samples to use for calibration during Block pruning."
    )
    parser.add_argument(
        "--num_calibration_samples_width",
        type=int,
        default=256,
        help="Number of samples to use for calibration during MLP channel pruning."
    )
    parser.add_argument(
        "--num_calibration_samples_ssm",
        type=int,
        default=256,
        help="Number of samples to use for calibration during SSM pruning."
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
    target_block_pruning_steps = args.target_block_pruning_steps
    target_width_pruning_steps = args.target_width_pruning_steps
    target_ssm_pruning_steps = args.target_ssm_pruning_steps
    mlp_channel_group_size = args.mlp_channel_group_size
    importance_metric = args.importance_metric
    calibration_dataset = args.calibration_dataset
    num_calibration_samples_block = args.num_calibration_samples_block
    num_calibration_samples_width = args.num_calibration_samples_width
    num_calibration_samples_ssm = args.num_calibration_samples_ssm
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16)
    total_params = sum(p.numel() for p in model.parameters())
    mamba_block_params = sum(p.numel() for n, p in model.named_parameters() if 'model.mamba_layers.0.' in n)
    mlp_block_params = sum(p.numel() for n, p in model.named_parameters() if 'model.blocks.0.feed_forward.' in n)
    mha_block_params = sum(p.numel() for n, p in model.named_parameters() if 'model.blocks.0.self_attn.' in n)
    model.eval()
    logging.info(f"Number of total parameters: {total_params}")
    logging.info(f"Number of Mamba block parameters: {mamba_block_params}")
    logging.info(f"Number of MHA block parameters: {mha_block_params}")
    logging.info(f"Number of MLP block parameters: {mlp_block_params}")

    logger.data["total_params"] = total_params
    logger.data["mamba_block_params"] = mamba_block_params
    logger.data["mlp_block_params"] = mlp_block_params
    logger.data["mha_block_params"] = mha_block_params

    num_layers = model.config.num_hidden_layers
    num_hybrid_layers = model.config.layers_block_type.count('g')

    mamba_block_ids = list(range(num_layers))
    mha_ids = list(range(num_hybrid_layers))
    mlp_ids = list(range(num_hybrid_layers))

    ssm_ids = list(range(num_layers))

    if do_prune:
        logging.info(f"Start pruning...")
        # calibration dataset
        dataset = utils.get_dataset(calibration_dataset)
        test_dataset = dataset["test"]
        test_dataset_block = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples_block))
        calibration_dataloader_block = utils.prepare_test_dataloader(
            dataset=test_dataset_block,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        test_dataset_width = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples_width))
        calibration_dataloader_width = utils.prepare_test_dataloader(
            dataset=test_dataset_width,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        test_dataset_ssm = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples_ssm))
        calibration_dataloader_ssm = utils.prepare_test_dataloader(
            dataset=test_dataset_ssm,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        importance_metric_func = utils.importance_metric_func_mapping[importance_metric]
        logging.info(
            f"Target pruning steps: "
            f"{target_block_pruning_steps} for Block (Mamba, MHA and MLP) pruning, "
            f"{target_width_pruning_steps} for MLP Width pruning, "
            f"{target_ssm_pruning_steps} for SSM pruning."
        )

        # init
        pruned_params = total_params
        logger.data["pruning_results"] = []
        num_evals = 0
        pruning_config = {
            "pruned_mamba_block_idx": [],
            "pruned_mha_idx": [],
            "pruned_mlp_idx": [],
            "pruned_mlp_channels": {i: model.config.ffn_hidden_size for i in range(num_hybrid_layers)},
            "pruned_ssm_idx": []
        }

        logging.info(f"Start Block (Mamba, MHA and MLP) pruning...")
        for step in range(target_block_pruning_steps):
            best_candidate = None
            lowest_importance = float('inf')

            # Mamba block
            pruned_target_idx = pruning_config["pruned_mamba_block_idx"]
            candidate_mamba_blocks = [target_id for target_id in mamba_block_ids if target_id not in pruned_target_idx]
            for target_id in tqdm(candidate_mamba_blocks, desc="Mamba Block"):
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
                    best_candidate = ("pruned_mamba_block_idx", target_id)
                    pruned_block_params = mamba_block_params

            # MHA
            pruned_target_idx = pruning_config["pruned_mha_idx"]
            candidate_mha_blocks = [target_id for target_id in mha_ids if target_id not in pruned_target_idx]
            for target_id in tqdm(candidate_mha_blocks, desc="MHA"):
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
                    best_candidate = ("pruned_mha_idx", target_id)
                    pruned_block_params = mha_block_params

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
                    pruned_block_params = mlp_block_params

            # pruning
            pruning_config[best_candidate[0]].append(best_candidate[1])
            prune(model, pruning_config)
            pruned_params -= pruned_block_params
            current_ratio = (1 - pruned_params / total_params) * 100

            config_path = os.path.join(pruning_config_path, f"config.block_pruning.{step}.json")
            with open(config_path, "w") as f:
                json.dump(pruning_config, f, indent=4)

            info = {
                "step": step,
                "pruning_target": re.search(r'pruned_(.*)_idx', best_candidate[0]).group(1),
                "importance": lowest_importance,
                "ratio": current_ratio,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs Block (Mamba, MHA and MLP) pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidate_mamba_blocks) + len(candidate_mlp_blocks) + len(candidate_mha_blocks)}")
            logging.info(f"Step {step} - Pruning Target: {best_candidate[0]}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Pruning Ratio: {current_ratio}")
            logging.info(f"Step {step} - Importance: {lowest_importance}")

        pruned_mlp_channels = pruning_config["pruned_mlp_channels"]
        logging.info(f"Start MLP width (intermediate layer) pruning...")

        mlp_groups = []
        for transformers_block in model.model.blocks:
            mlp_groups.append(replace_with_mask_linear(transformers_block.feed_forward))

        for step in range(target_block_pruning_steps, target_block_pruning_steps + target_width_pruning_steps):
            best_candidate = None
            lowest_channel_groups_importance = float('inf')
            candidates = []
            for mlp_id, width in pruned_mlp_channels.items():
                if (width - mlp_channel_group_size) > 0 and (mlp_id not in pruning_config["pruned_mlp_idx"]):
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
                    dataloader=calibration_dataloader_width,
                    pad_token_id=tokenizer.eos_token_id
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

            pruned_mlp_width_params = mlp_channel_group_size * model.config.hidden_size * 3
            pruned_params -= pruned_mlp_width_params
            current_ratio = (1 - pruned_params / total_params) * 100

            config_path = os.path.join(pruning_config_path, f"config.mlp_width.{step}.json")
            pruning_config_str = {str(dim): value for dim, value in pruning_config.items()}
            with open(config_path, "w") as f:
                json.dump(pruning_config_str, f, indent=4)

            info = {
                "step": step,
                "pruning_target": "mlp_width",
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

        logging.info(f"Start SSM pruning...")
        for step in range(
                target_block_pruning_steps + target_width_pruning_steps,
                target_block_pruning_steps + target_width_pruning_steps + target_ssm_pruning_steps
        ):
            best_candidate = None
            lowest_ssm_importance = float('inf')
            candidate_ssms = [
                ssm_id for ssm_id in ssm_ids
                if ssm_id not in pruning_config["pruned_ssm_idx"]
                and ssm_id not in pruning_config["pruned_mamba_block_idx"]
            ]

            for ssm_id in tqdm(candidate_ssms):
                num_evals += 1
                pruning_config["pruned_ssm_idx"].append(ssm_id)
                prune(model, pruning_config)

                # the importance of the current SSM
                ssm_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader_ssm,
                    pad_token_id=tokenizer.eos_token_id
                )

                pruning_config["pruned_ssm_idx"] = pruning_config["pruned_ssm_idx"][:-1]
                prune(model, pruning_config)

                # expect to prune the least important SSM
                if ssm_importance < lowest_ssm_importance:
                    lowest_ssm_importance = ssm_importance
                    best_candidate = ssm_id

            # pruning
            pruning_config["pruned_ssm_idx"].append(best_candidate)
            prune(model, pruning_config)
            config_path = os.path.join(pruning_config_path, f"config.ssm.{step}.json")
            with open(config_path, "w") as f:
                json.dump(pruning_config, f, indent=4)

            info = {
                "step": step,
                "pruning_target": "ssm",
                "importance": lowest_ssm_importance,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} performs SSM pruning")
            logging.info(f"Step {step} - Number of Candidates: {len(candidate_ssms)}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Importance: {lowest_ssm_importance}")

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

        mlp_groups = []
        for transformers_block in model.model.blocks:
            mlp_groups.append(replace_with_mask_linear(transformers_block.feed_forward))
        prune(model, pruned_config, groups_for_width_pruning=mlp_groups)
        logging.info(f"Detect a pruned model config: {pruned_config}")

        # Evaluate on selected tasks
        mamba_lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        logging.info(f"Selected Tasks: {TASKS}")

        results = evaluator.simple_evaluate(mamba_lm, tasks=TASKS, batch_size=batch_size, log_samples=False)['results']

        metric_vals = {}
        total_score = 0.0
        for task, result in results.items():
            res = result['acc,none'] if task == 'arc_easy' else result.get('acc_norm,none', result['acc,none'])
            total_score += res
            metric_vals[task] = round(res, 3) * 100
            if task == "lambada_openai":
                metric_vals[task + "_ppl"] = result['perplexity,none']
        logging.info(json.dumps(metric_vals, indent=4))

        n_tasks = len(TASKS)
        acc_avg = round(total_score / n_tasks, 3) * 100
        logging.info(f"Average accuracy across tasks: {acc_avg}")

        # Save evaluation results
        overall_results = {
            "total_params": total_params,
            "7cs_acc_avg": acc_avg,
            **metric_vals
        }
        result_file_name = ".".join(pruned_model_config_file.split("/")[-1].split(".")[:-1])
        eval_result_path = os.path.join(output_path, f"eval.res.{result_file_name}.json")
        with open(eval_result_path, "w") as f:
            json.dump(overall_results, f, indent=4)

        eval_result_csv_path = os.path.join(output_path, f"eval.res.{result_file_name}.csv")
        columns = ["total_params"] + ["lambada_openai_ppl"] + TASKS + ["7cs_acc_avg"]
        with open(eval_result_csv_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow(overall_results)

        logger.data[f"evaluation_{result_file_name}"] = overall_results
        logger.update()

if __name__ == "__main__":
    main()
