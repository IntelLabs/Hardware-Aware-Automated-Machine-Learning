import os
import argparse
import json
import logging
import time
import sys

from transformers import set_seed

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from t2l_config import add_t2l_args, parse_t2l_args
from models.t2l_model import load_t2l_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="coqa",
        choices=["coqa", "gsm8k"],
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit for the number of evaluation samples."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory to save the results."
    )
    add_t2l_args(parser)
    args = parser.parse_args()
    task = args.task
    limit = args.limit
    output_path = args.output_path
    t2l_config = parse_t2l_args(args)
    set_seed(42)

    # Generate result folder name
    result_folder_name = t2l_config.generate_result_folder_name(task, limit)
    output_path = os.path.join(output_path, result_folder_name)

    result_file_path = os.path.join(output_path, 'result.json')
    if os.path.exists(result_file_path):
        print(f"The file {result_file_path} already exists. Exiting the program.")
        sys.exit(1)
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    model = load_t2l_model(t2l_config)

    logging.info(f"Start evaluation...")
    hflm = HFLM(pretrained=model, tokenizer=model.tokenizer, batch_size=1)
    logging.info(f"Selected Tasks: {task}")
    if limit == -1:
        limit = None

    start_time = time.time()
    lm_eval_outputs = evaluator.simple_evaluate(hflm, tasks=task, num_fewshot=0, batch_size=1, log_samples=True, limit=limit)
    end_time = time.time()

    # Save results
    samples = lm_eval_outputs["samples"]
    with open(os.path.join(output_path, "log_samples.json"), "w") as f:
        json.dump(samples, f, indent=4)

    results = lm_eval_outputs['results']
    results["total_time_taken"] = end_time - start_time
    results["config"] = vars(args)
    logging.info(json.dumps(results, indent=4))

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
