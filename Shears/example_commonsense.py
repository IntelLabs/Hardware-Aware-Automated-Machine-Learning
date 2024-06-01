import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Response:
                    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default="shears-llama-7b-50-base", type=str)
    parser.add_argument("--adapter_model_path", default="IntelLabs/shears-llama-7b-50-cs-heuristic-adapter", type=str)
    args = parser.parse_args()
    base_model_path = args.base_model_path
    adapter_model_path = args.adapter_model_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    non_zero_params = sum([(param.data != 0).sum().item() for _, param in model.named_parameters()])
    print(f"Number of all non-zero parameters: {non_zero_params}")

    instructions = [
        "Please choose the correct answer to the question: A cactus stem is used to store\n\nAnswer1: fruit "
        "Answer2: liquid Answer3: food Answer4: spines\n\nAnswer format: answer1/answer2/answer3/answer4",

        "Please choose the correct solution to the question: Prevent bottles from rolling in fridge.\n\n"
        "Solution1: Put binder clip on fridge shelves to prevent sliding.\n\nSolution2: Put staple remover on "
        "fridge shelves to prevent sliding.\n\nAnswer format: solution1/solution2",

        "Please choose the correct answer to the question: Which characteristic describes the texture of a "
        "kitten's fur?\n\nAnswer1: gray Answer2: warm Answer3: long Answer4: soft\n\nAnswer format: answer1/"
        "answer2/answer3/answer4",
    ]

    for idx, instruction in enumerate(instructions):
        print(f"Example {idx}:")
        prompt = generate_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                use_cache=True,
                num_beams=4,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)


if __name__ == "__main__":
    main()
