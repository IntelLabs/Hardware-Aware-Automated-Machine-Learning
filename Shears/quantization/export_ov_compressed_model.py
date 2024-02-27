# Adapted from https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
import argparse
import os

import torch
from optimum.intel.openvino import OVModelForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM

import nncf


def check_sparsity(model):
    """
    Only consider the sparsity of all layers
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
    layers = model.model.layers

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default="yahma/llama-7b-hf", type=str)
    parser.add_argument("--adapter_path", default=None, type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--output_dir", default="llama7b-ov-compressed", type=str)

    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    adapter_path = args.adapter_path
    compress = args.compress
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # check sparsity and non-zero params.
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if adapter_path is not None:
        PeftModel.from_pretrained(model, adapter_path)
    sparsity = check_sparsity(model)
    print(f"Sparsity: {sparsity}")
    non_zero_params = sum([(param.data != 0).sum().item() for _, param in model.named_parameters()])
    print(f"Number of all non-zero parameters: {non_zero_params}")

    model = OVModelForCausalLM.from_pretrained(
        model_name_or_path, adapter_path=adapter_path, export=True, load_in_8bit=False, compile=False, stateful=False
    )
    if compress:
        model.model = nncf.compress_weights(model.model)
        print("Compressed")
    model.save_pretrained(output_dir)
    print(f"Saved in {output_dir}")


if __name__ == "__main__":
    main()
