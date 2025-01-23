import argparse
import json
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

TASKS = ["arc_easy", "arc_challenge", "piqa", "winogrande", "hellaswag"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    args = parser.parse_args()
    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype="float16",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=64)

    # Evaluate on selected tasks
    logging.info(f"Selected Tasks: {TASKS}")
    results = evaluator.simple_evaluate(lm, tasks=TASKS, num_fewshot=0, batch_size=64, log_samples=False)['results']

    metric_vals = {}
    for task, result in results.items():
        res = result['acc,none']
        metric_vals[task] = round(res, 3) * 100

    logging.info(json.dumps(metric_vals, indent=4))


if __name__ == "__main__":
    main()
