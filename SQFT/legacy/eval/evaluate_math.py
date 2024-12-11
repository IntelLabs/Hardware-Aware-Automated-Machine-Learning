import argparse
import copy
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel


DATASETS = ["mawps", "SVAMP", "gsm8k"]

def extract_answer_number(dataset_name: str, sentence: str) -> float:
    """
    Extracts the numerical answer from a given sentence based on the dataset type.

    Args:
        dataset_name (str): The name of the dataset.
        sentence (str): The sentence from which to extract the answer.

    Returns:
        float: The extracted numerical answer.
    """
    dataset_name = dataset_name.lower()
    if dataset_name in ["gsm8k", "svamp", "mawps"]:
        sentence = sentence.replace(",", "")
        predictions = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
        if not predictions:
            return float("inf")
        predicted_answer = float(predictions[-1])
    else:
        raise NotImplementedError(f"Dataset not supported: {dataset_name}")

    if isinstance(predicted_answer, str):
        try:
            predicted_answer = float(predicted_answer)
        except ValueError:
            predicted_answer = float("inf")

    return predicted_answer


def load_test_data(test_dataset: str) -> list:
    """
    Loads test data from a specified dataset file.

    Args:
        test_dataset (str): The name of the test dataset.

    Returns:
        list: The loaded test data.
    """
    file_path = f"datasets/{test_dataset}/test.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")

    with open(file_path, "r") as file:
        json_data = json.load(file)

    return json_data


def generate_prompt_eval(instruction: str, model_type : str) -> str:
    """
    Generates a prompt for evaluation based on the given instruction.

    Args:
        instruction (str): The instruction to include in the prompt.

    Returns:
        str: The generated prompt.
    """
    if model_type == "phi3":
        return f"<|user|>\n{instruction} <|end|>\n<|assistant|>"
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                            ### Instruction:
                            {instruction}

                            ### Response:
                            """


def evaluate_one_sample(
    instruction: str,
    model,
    tokenizer,
    num_beams: int = 4,
    max_new_tokens: int = 32,
) -> str:
    """
    Evaluates a single sample using the model and tokenizer.

    Args:
        instruction (str): The instruction to evaluate.
        model: The model to use for evaluation.
        tokenizer: The tokenizer to use for evaluation.
        num_beams (int, optional): The number of beams for beam search. Defaults to 4.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 32.

    Returns:
        str: The generated output.
    """
    model_type = model.config.model_type
    prompt = generate_prompt_eval(instruction, model_type=model_type)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            num_beams=num_beams,
        )

    sequences = generation_output.sequences[0]
    output = tokenizer.decode(sequences)
    answer_key = "<|assistant|>" if model_type == "phi3" else "### Response:"
    return output.split(answer_key)[1].strip()


def evaluate(model, tokenizer, dataset_name: str, save_file: str) -> float:
    """
    Evaluates the model on a specified dataset and saves the results.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use for evaluation.
        dataset_name (str): The name of the dataset to evaluate on.
        save_file (str): The file to save the evaluation results.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    dataset = load_test_data(dataset_name)
    total_samples = len(dataset)

    correct_predictions = 0
    tolerance = 0.001
    output_data = []

    for index, data in tqdm(enumerate(dataset)):
        instruction = data.get("instruction")
        output = evaluate_one_sample(instruction, model=model, tokenizer=tokenizer, max_new_tokens=256)
        label = data.get("answer")
        is_correct = False

        if isinstance(label, str):
            label = float(label)
        prediction = extract_answer_number(dataset_name, output)
        if abs(label - prediction) <= tolerance:
            correct_predictions += 1
            is_correct = True

        new_data = copy.deepcopy(data)
        new_data["output_pred"] = output
        new_data["pred"] = prediction
        new_data["flag"] = is_correct
        output_data.append(new_data)

        print(output)
        print(f"Prediction: {prediction}")
        print(f"Label: {label}")
        print(
            f"\rTest: {index + 1}/{total_samples} | Accuracy: {correct_predictions}  {correct_predictions / (index + 1)}")

    with open(save_file, "w") as file:
        json.dump(output_data, file, indent=4)

    accuracy = correct_predictions / total_samples
    return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, help="Path to the base model.")
    parser.add_argument("--tokenizer_path", default=None, type=str, help="Path to the tokenizer.")
    parser.add_argument("--dtype", default="float16", type=str, help="Data type for the model.")
    parser.add_argument("--adapter_model_path", type=str, help="Path to the adapter model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions will be written.",
    )
    args = parser.parse_args()

    base_model_path = args.base_model_path
    tokenizer_path = args.tokenizer_path
    dtype = args.dtype
    adapter_model_path = args.adapter_model_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if tokenizer_path is None else tokenizer_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if adapter_model_path is not None:
        model = PeftModel.from_pretrained(model, adapter_model_path, torch_dtype=dtype, device_map="auto")

    model.eval()

    result_file_path = os.path.join(output_dir, "result.json")
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as file:
            dataset_to_accuracy = json.load(file)
    else:
        dataset_to_accuracy = {}

    for dataset_name in DATASETS:
        if dataset_name in dataset_to_accuracy:
            continue
        print(f"*** Evaluation on {dataset_name} ***")
        save_file = os.path.join(output_dir, f"predict_{dataset_name}.json")
        accuracy = evaluate(model, tokenizer, dataset_name, save_file)
        dataset_to_accuracy[dataset_name] = accuracy
        print(f"{dataset_name} - Accuracy: {accuracy}")

    accuracies = [acc for dataset_name, acc in dataset_to_accuracy.items() if dataset_name in DATASETS]
    average_accuracy = sum(accuracies) / len(accuracies)
    dataset_to_accuracy["average"] = average_accuracy
    with open(result_file_path, "w") as file:
        json.dump(dataset_to_accuracy, file, indent=4)

    print(dataset_to_accuracy)


if __name__ == "__main__":
    main()
