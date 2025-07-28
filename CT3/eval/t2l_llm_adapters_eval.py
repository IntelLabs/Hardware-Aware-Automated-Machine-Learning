# This script is based on modifications from 
# https://github.com/AGI-Edgerunners/LLM-Adapters/blob/816657208af4db747803f87ba40a4c71383fed7a/evaluate.py
import os
import argparse
import json
import logging
import time
import re
import random
import sys
from tqdm import tqdm

import torch
from transformers import set_seed

from t2l_config import add_t2l_args, parse_t2l_args
from models.t2l_model import load_t2l_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="mawps",
        choices=["mawps", "mathqa"],
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
    
    def extract_answer_number(sentence: str) -> float:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float('inf')
        return pred_answer

    def extract_answer_letter(sentence: str) -> str:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'A|B|C|D|E', sentence_)
        if pred_answers:
            if not pred_answers:
                return ''
            return pred_answers[0]
        else:
            return ''

    def evaluate(
        instruction,
        max_new_tokens=1024,
    ):
        prompt = instruction
        inputs = model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        s = generation_output.sequences[0]
        input_length = input_ids.shape[1]
        response_ids = s[input_length:]
        output = model.tokenizer.decode(response_ids, skip_special_tokens=True)
        return output

    file_path = f'llm_adapters_test_datasets/{task}/test.json'
    test_dataset = json.load(open(file_path, 'r'))
    if limit != -1:
        random.seed(42)
        random.shuffle(test_dataset)
        test_dataset = test_dataset[:limit]
    correct = 0
    miss = 0.001
    total = len(test_dataset)
    start_time = time.time()
    pbar = tqdm(total=total)
    for idx, data in enumerate(test_dataset):
        instruction = data.get('instruction')
        response = evaluate(instruction)
        label = data.get('answer')
        flag = False
        if task in ["mathqa", "AQuA"]:
            predict = extract_answer_letter(response)
            if label == predict:
                correct += 1
                flag = True
        else:
            if isinstance(label, str):
                label = float(label)
            predict = extract_answer_number(response)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        # print(f"instruction: {instruction}")
        # print(f"response: {response}")
        # print(f"Is the answer correct?: {flag}")
        # if (idx + 1) % 10 == 0:
        #     print(f"Accuracy: {correct / (idx + 1)}")
        pbar.update(1)
    pbar.close()
    end_time = time.time()

    # Save results
    results = {
        "Accuracy": correct / total,
        "total_time_taken": end_time - start_time,
        "config": vars(args)
    }
    logging.info(json.dumps(results, indent=4))

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
