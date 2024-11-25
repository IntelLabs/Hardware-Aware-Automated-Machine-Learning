import argparse
import importlib.util
import json
import os
from pathlib import Path

from tabulate import tabulate
from transformers import AutoModelForCausalLM

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import load_yaml_config

from nncf import NNCFConfig
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.torch.model_creation import create_nncf_network
from peft import PeftModel


# Get the directory path where the `lm_eval` module is located
spec = importlib.util.find_spec("lm_eval")
module_path = spec.origin
module_dir = os.path.dirname(module_path)

arc_yaml_file = os.path.join(module_dir, "tasks/arc/arc_easy.yaml")
task_config = load_yaml_config(arc_yaml_file)
# Modify the task configuration to define the validation set task
task_config["task"] = "arc_easy_val"
task_config["dataset_name"] = "ARC-Easy"
task_config["test_split"] = "validation"


def main(base_model_path, adapter_model_path, output_dir, num_evals=100, population=5):
    os.makedirs(output_dir, exist_ok=True)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_model_path, torch_dtype="float32")
    for param in model.parameters():
        param.requires_grad = True
    nncf_config = os.path.join(adapter_model_path, "nncf_config.json")
    with safe_open(Path(nncf_config)) as file:
        loaded_json = json.load(file)
    loaded_json["bootstrapNAS"] = loaded_json["SQFT"]
    del loaded_json["SQFT"]
    loaded_json["bootstrapNAS"]["training"]["algorithm"] = "progressive_shrinking"
    loaded_json["bootstrapNAS"]["training"]["batchnorm_adaptation"] = {"num_bn_adaptation_samples": 0}
    loaded_json["bootstrapNAS"]["search"] =  {
        "algorithm": "NSGA2",
        "batchnorm_adaptation": {
            "num_bn_adaptation_samples": 0
        },
        "num_evals": num_evals,
        "population": population,
        "ref_acc": -1,
        "acc_delta": 0.01
    }
    nncf_config = NNCFConfig.from_dict(loaded_json)
    nncf_config["log_dir"] = output_dir
    model = create_nncf_network(model, nncf_config)
    for param in model.parameters():
        param.requires_grad = False
    compression_ctrl, model = create_compressed_model_from_algo_names(
        model, nncf_config, algo_names=["progressive_shrinking"]
    )

    compression_ctrl.multi_elasticity_handler.enable_all()
    search_algo = BaseSearchAlgorithm.from_config(model, compression_ctrl, nncf_config)
    lm = HFLM(model, batch_size=32, trust_remote_code=True)
    task_manager = TaskManager("INFO", include_path=None)
    request_caching_args = {'cache_requests': False, 'delete_requests_cache': False, 'rewrite_requests_cache': False}

    def validate_fn(model_, eval_dataset):
        # Evaluate the current sub-adapter configuration
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=[task_config],
            batch_size=32,
            log_samples=False,
            task_manager=task_manager,
            **request_caching_args,
        )
        accuracy = results["results"]["arc_easy_val"]["acc_norm,none"]
        print(f"Sub-adapter Accuracy: {accuracy}")
        return accuracy

    # Test Heuristic and Minimal sub-adapter on the validation dataset
    compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
    heuristic_config = {k: v[(len(v) - 1) // 2] for k, v in compression_ctrl.multi_elasticity_handler.width_search_space.items()}
    heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
    compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
    heu_eval_acc = validate_fn(model, None)
    compression_ctrl.multi_elasticity_handler.activate_minimum_subnet()
    min_eval_acc = validate_fn(model, None)

    elasticity_ctrl, best_config, performance_metrics = search_algo.run(
        validate_fn, None, output_dir
    )

    search_algo.search_progression_to_csv()
    search_algo.evaluators_to_csv()
    search_algo.visualize_search_progression()

    sorted_config = sorted(search_algo.search_records, key=lambda x: x[-1])
    best_configs = sorted_config[:3]

    data = [
        ["Heuristic", heu_eval_acc]
    ]
    for i, config in enumerate(best_configs):
        data.append([f"Top{i + 1} Best", -config[-1]])
    headers = ["Sub-adapter", "Arc-Easy Accuracy (validation)"]
    markdown_table = tabulate(data, headers, tablefmt="github")
    print(markdown_table)

    # Test
    def test_fn():
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=["arc_easy"],
            batch_size=32,
            log_samples=False,
            task_manager=task_manager,
            **request_caching_args,
        )
        arc_easy_acc = results["results"]["arc_easy"]["acc_norm,none"]
        return arc_easy_acc


    # Heuristic
    compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
    compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
    heu_test_acc = test_fn()

    # Minimal
    compression_ctrl.multi_elasticity_handler.activate_minimum_subnet()
    min_test_acc = test_fn()

    data = [
        ["Heuristic", heu_eval_acc, heu_test_acc],
        ["Minimal", min_eval_acc, min_test_acc],
    ]
    for i, config in enumerate(best_configs):
        compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(config[0])
        print(config[0])
        test_acc = test_fn()
        data.append([f"Top{i + 1} Best", -config[-1], test_acc])
    headers = ["Sub-adapter", "Arc-Easy Accuracy (validation)", "Arc-Easy Accuracy (test)"]
    markdown_table = tabulate(data, headers, tablefmt="github")
    print(markdown_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge base model and adapter model")
    parser.add_argument('--base_model_path', type=str, required=True, help="Path to the base model")
    parser.add_argument('--adapter_model_path', type=str, required=True, help="Path to the adapter model")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_evals', type=int, default=200)
    parser.add_argument('--population', type=int, default=5)

    args = parser.parse_args()
    main(args.base_model_path, args.adapter_model_path, args.output_dir, args.num_evals, args.population)
