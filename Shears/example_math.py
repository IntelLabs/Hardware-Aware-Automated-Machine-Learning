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
    parser.add_argument("--model_path", default="llama-7b-sparsity50-shears-math-heuristic", type=str)
    args = parser.parse_args()
    model_path = os.path.join(args.model_path, "base_model")
    lora_weights = os.path.join(args.model_path, "adapter_model")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_weights, torch_dtype=torch.float16, device_map={"": 0})

    non_zero_params = sum([(param.data != 0).sum().item() for _, param in model.named_parameters()])
    print(f"Number of all non-zero parameters: {non_zero_params}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    model.eval()

    instructions = [
        "Jack had $100. Sophia gave him 1/5 of her $100. How many dollars does Jack have now?",
        "Edgar eats 18 pretzels a day. If his brother eats 1/2 as many, how many does his brother eat in a week?",
        "Trent is 5 years older than Jane, and Jane is 3 years younger than Quinn. If Quinn is 30, how old is Trent?",
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
