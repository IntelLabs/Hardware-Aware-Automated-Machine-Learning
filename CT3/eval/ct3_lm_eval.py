import os
import argparse
import json
import logging
import time
import sys
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

try:
    from ipex_llm.transformers import AutoModelForCausalLM as IpexAutoModelForCausalLM
except ImportError:
    print("ipex-llm is not installed. Please install ipex-llm to use this feature.")

from ct3_config import add_ct3_args, parse_ct3_args
from models.model import convert_to_ttt_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pre-trained model.",
        required=True
    )
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
    add_ct3_args(parser)
    args = parser.parse_args()
    model_path = args.model_path
    task = args.task
    limit = args.limit
    output_path = args.output_path
    ct3_config = parse_ct3_args(args)

    set_seed(42)

    # Generate result folder name
    result_folder_name = ct3_config.generate_result_folder_name(task, limit)
    output_path = os.path.join(output_path, result_folder_name)

    result_file_path = os.path.join(output_path, 'result.json')
    if os.path.exists(result_file_path):
        print(f"The file {result_file_path} already exists. Exiting the program.")
        sys.exit(1)
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if ct3_config.use_ipex_llm:
        model = IpexAutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_low_bit="bf16",
            optimize_model=False,
            torch_dtype=torch.bfloat16,
            # modules_to_not_convert=["lm_head"],
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        ).eval()
    
    model = convert_to_ttt_model(model, tokenizer, ct3_config)

    logging.info(f"Start evaluation...")
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
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

    # save analysis result
    if ct3_config.ttt and ct3_config.analysis:
        model.save_analysis_data(output_path)


if __name__ == "__main__":
    main()
