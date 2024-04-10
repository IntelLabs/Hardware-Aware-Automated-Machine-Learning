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
    parser.add_argument("--model_path", default="llama-7b-sparsity50-shears-commonsense-heuristic", type=str)
    args = parser.parse_args()
    model_path = os.path.join(args.model_path, "base_model")
    lora_weights = os.path.join(args.model_path, "adapter_model")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, lora_weights, torch_dtype=torch.float16, device_map={"": 0})

    non_zero_params = sum([(param.data != 0).sum().item() for _, param in model.named_parameters()])
    print(f"Number of all non-zero parameters: {non_zero_params}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    model.eval()

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
