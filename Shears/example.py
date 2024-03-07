# Adapted from https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer


def check_sparsity(model):
    """
    Consider the sparsity of all layers
    Refer to https://github.com/locuslab/wanda/blob/main/lib/prune.py#L32.
    """

    def find_layers(module, name=""):
        """
        Recursively find the layers of a certain type in a module.

        Args:
            module (nn.Module): PyTorch module.
            layers (list): List of layer types to find.
            name (str): Name of the module.

        Returns:
            dict: Dictionary of layers of the given type(s) within the module.
        """
        res = {}
        if isinstance(module, torch.nn.Linear):
            res[name] = module
        for name1, child in module.named_children():
            res.update(find_layers(child, name=name + "." + name1 if name != "" else name1))
        return res

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.base_model.model.model.layers  # LLaMA
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            W = subset[name].weight.data
            count += (W != 0).sum().item()
            total_params += W.numel()
    model.config.use_cache = use_cache
    return 1 - float(count) / total_params


def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                    ### Instruction:
                    {instruction}

                    ### Response:
                    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llama-7b-sparsity50-shears-math", type=str)
    args = parser.parse_args()
    model_path = os.path.join(args.model_path, "pretrained_weights")
    lora_weights = os.path.join(args.model_path, "adapter_weights")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, lora_weights, torch_dtype=torch.float16, device_map={"": 0})

    sparsity = check_sparsity(model)
    print(f"Sparsity: {sparsity}")
    non_zero_params = sum([(param.data != 0).sum().item() for _, param in model.named_parameters()])
    print(f"Number of all non-zero parameters: {non_zero_params}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0

    instructions = [
        "Jack had $100. Sophia gave him 1/5 of her $100. How many dollars does Jack have now?",
        "Edgar eats 18 pretzels a day. If his brother eats 1/2 as many, how many does his brother eat in a week?",
        "Trent is 5 years older than Jane, and Jane is 3 years younger than Quinn. If Quinn is 30, how old is Trent?"
    ]
    for idx, instruction in enumerate(instructions):
        print(f"Example {idx}:")
        prompt = generate_prompt(instruction)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256,
                use_cache=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)


if __name__ == "__main__":
    main()
