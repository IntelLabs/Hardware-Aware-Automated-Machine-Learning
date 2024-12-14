import os
import json
import logging
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    model_path = args.model_path
    output_path = args.output_path

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="float16",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=64)

    task_names = ["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"]
    logging.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=64, log_samples=False)['results']

    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) * 100 for task, result in results.items()}
    logging.info(json.dumps(metric_vals, indent=4))

    def calculate_avg_accuracy(task_names, results):
        n_tasks = len(task_names)
        acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
        return round(acc_cumul / n_tasks, 4) * 100

    acc_avg = calculate_avg_accuracy(task_names, results)
    logging.info(f"Average accuracy across tasks: {acc_avg}")

    # Save evaluation results
    overall_results = {
        "ppl_wikitext2": dataset_ppl,
        "5cs_acc_avg": acc_avg,
        **metric_vals
    }
    eval_result_path = os.path.join(output_path, f"eval.res.json")
    with open(eval_result_path, "w") as f:
        json.dump(overall_results, f, indent=4)


if __name__ == "__main__":
    main()
