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

TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]


def mask_mamba_block(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mask_mamba_block")
        if idx in layer_indices:
            layers[idx].mask_mamba_block = True
        else:
            layers[idx].mask_mamba_block = False


def mask_ssm(layers, layer_indices):
    for idx in range(len(layers)):
        assert hasattr(layers[idx], "mamba") and hasattr(layers[idx].mamba, "mask_ssm")
        if idx in layer_indices:
            layers[idx].mamba.mask_ssm = True
        else:
            layers[idx].mamba.mask_ssm = False


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
        help="Path to Zamba model."
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
        "--prune_target",
        choices=["mamba_block", "ssm"],
        default="mamba_block",
    )
    parser.add_argument(
        "--target_pruning_steps",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--importance_metric",
        type=str,
        default="ppl",
        choices=["ppl"],
        help="Metric for calculating Mamba-block/SSM importance."
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
        help="Number of samples to use for calibration during Mamba-block/SSM pruning."
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
    prune_target = args.prune_target
    target_pruning_steps = args.target_pruning_steps
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
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype=torch.float16)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of total parameters: {total_params}")
    logger.data["total_params"] = total_params
    num_layers = model.config.num_hidden_layers
    num_hybrid_layers = model.config.layers_block_type.count('g')
    target_ids = list(range(num_layers))

    if do_prune:
        logging.info(f"Start pruning...")
        mask_func = mask_mamba_block if prune_target == "mamba_block" else mask_ssm
        # calibration dataset
        dataset = utils.get_dataset(calibration_dataset)
        test_dataset = dataset["test"]
        test_dataset = test_dataset.select(random.sample(range(len(test_dataset)), num_calibration_samples))
        calibration_dataloader = utils.prepare_test_dataloader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=2
        )
        importance_metric_func = utils.importance_metric_func_mapping[importance_metric]
        logging.info(f"Target pruning steps : {target_pruning_steps}")

        step = 0
        pruning_config = []
        logger.data["pruning_results"] = []
        num_evals = 0

        logging.info(f"Start pruning...")
        while True:
            best_candidate = None
            lowest_importance = float('inf')
            candidate_targets = [target_id for target_id in target_ids if target_id not in pruning_config]

            for target_id in tqdm(candidate_targets):
                num_evals += 1
                pruning_config.append(target_id)
                mask_func(model.model.mamba_layers, pruning_config)

                # the importance of the current Mamba-block/SSM
                target_importance = importance_metric_func(
                    model=model,
                    dataloader=calibration_dataloader,
                    pad_token_id=tokenizer.eos_token_id
                )
                pruning_config = pruning_config[:-1]

                # expect to prune the least important Mamba-block/SSM
                if target_importance < lowest_importance:
                    lowest_importance = target_importance
                    best_candidate = target_id

            # pruning
            pruning_config.append(best_candidate)
            config_path = os.path.join(pruning_config_path, f"config.{prune_target}.{step}.json")
            with open(config_path, "w") as f:
                json.dump({
                    f"pruned_{prune_target}_idx": pruning_config
                }, f, indent=4)

            info = {
                "step": step,
                "prune_target": prune_target,
                "importance": lowest_importance,
                "config_save_to": config_path,
            }
            logger.data["pruning_results"].append(info)
            logger.update()

            logging.info(f"Step {step} - Number of Candidates: {len(candidate_targets)}")
            logging.info(f"Step {step} - Config: {pruning_config}")
            logging.info(f"Step {step} - Importance: {lowest_importance}")
            step += 1
            if step == target_pruning_steps:
                break

        logger.data["num_evals"] = num_evals
        logger.update()

        # save the last pruning config
        config_path = os.path.join(output_path, f"pruning_config.json")
        with open(config_path, "w") as f:
            json.dump({
                f"pruned_{prune_target}_idx": pruning_config
            }, f, indent=4)

    if do_eval:
        logging.info(f"Start evaluation...")

        if pruned_model_config_file is None:
            assert do_prune
            pruned_model_config_file = config_path

        # Load pruning results
        with open(pruned_model_config_file, "r") as f:
            pruned_config = json.load(f)
        if "pruned_mamba_block_idx" in pruned_config:
            mask_mamba_block(model.model.mamba_layers, pruned_config["pruned_mamba_block_idx"])
        if "pruned_ssm_idx" in pruned_config:
            mask_ssm(model.model.mamba_layers, pruned_config["pruned_ssm_idx"])
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
