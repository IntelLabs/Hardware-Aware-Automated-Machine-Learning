import argparse
import torch

from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
from transformers import AutoModelForCausalLM


def de_quantize(quant_linear_module):
    """
    De-quantize the weights of a QuantLinear module.

    This function is referenced from:
        https://github.com/AutoGPTQ/AutoGPTQ/blob/866b4c8c2cbb893f1156cb6c114625bba2e4d7c5/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py#L296-L316

    Args:
        quant_linear_module (QuantLinear): The QuantLinear module to de-quantize.

    Returns:
        torch.Tensor: The de-quantized weights.
    """
    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(quant_linear_module.qzeros, 2).expand(-1, -1, 32 // quant_linear_module.bits),
        quant_linear_module.wf.unsqueeze(0),
    ).to(torch.int16 if quant_linear_module.bits == 8 else torch.int8)

    zeros = zeros + 1
    zeros = torch.bitwise_and(
        zeros, (2 ** quant_linear_module.bits) - 1
    )  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.

    zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

    scales = quant_linear_module.scales
    scales = scales.reshape(-1, 1, scales.shape[-1])

    weight = torch.bitwise_right_shift(
        torch.unsqueeze(quant_linear_module.qweight, 1).expand(-1, 32 // quant_linear_module.bits, -1),
        quant_linear_module.wf.unsqueeze(-1),
    ).to(torch.int16 if quant_linear_module.bits == 8 else torch.int8)
    weight = torch.bitwise_and(weight, (2 ** quant_linear_module.bits) - 1)
    weight = weight.reshape(-1, quant_linear_module.group_size, weight.shape[2])
    weight = scales * (weight - zeros)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    return weight


def find_layers(module, layers=[torch.nn.Linear, QuantLinear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the sparsity of the model's layers.

    Args:
        model (nn.Module): The model to check sparsity for.

    Returns:
        float: The sparsity of the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i, layer in enumerate(layers):
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            if isinstance(subset[name], QuantLinear):
                W = de_quantize(subset[name])
            else:
                W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"Layer {i} sparsity: {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    args = parser.parse_args()
    model_path = args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).eval()

    print(f"Sparsity: {check_sparsity(model)}")


if __name__ == "__main__":
    main()
