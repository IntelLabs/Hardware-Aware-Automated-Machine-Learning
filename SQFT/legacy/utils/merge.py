import argparse
import sys
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('.')


def main():
    parser = argparse.ArgumentParser(description="Merge base model and adapter model, then save the merged model and tokenizer.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer. Defaults to None.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model. Defaults to 'float16'.")
    parser.add_argument("--non_quant_base_model_path", type=str, default=None, help="Path to the non-quantized base model.")
    parser.add_argument("--adapter_model_path", type=str, required=True, help="Path to the adapter model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model and tokenizer.")
    args = parser.parse_args()

    base_model_path = args.base_model_path
    tokenizer_path = args.tokenizer_path
    dtype = args.dtype
    non_quant_base_model_path = args.non_quant_base_model_path
    adapter_model_path = args.adapter_model_path
    output_path = args.output_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_model_path,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()

    if getattr(model.active_peft_config, "quantization_aware", False):
        from modules.sqft_linear import SQFTQuantAwareLinear

        def find_layers(module, layers=[SQFTQuantAwareLinear], name=''):
            """
            Recursively find layers of a specific type in a module.
            """
            if type(module) in layers:
                return {name: module}
            res = {}
            for child_name, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + child_name if name != '' else child_name
                ))
            return res

        non_quant_model = None
        non_quant_model_modules = None
        if non_quant_base_model_path is not None:
            non_quant_model = AutoModelForCausalLM.from_pretrained(
                non_quant_base_model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            non_quant_model_modules = {name: module for name, module in non_quant_model.model.layers.named_modules()}

        # Initialize the quantization layer using full/half-precision weight parameters
        sqft_quant_aware_layers = find_layers(model.base_model.model.model.layers)
        for layer_name in sqft_quant_aware_layers:
            sqft_quant_aware_layer = sqft_quant_aware_layers[layer_name]
            custom_weight = None
            if non_quant_model_modules is not None:
                custom_weight = non_quant_model_modules[layer_name].weight.data
            sqft_quant_aware_layer.init_params(custom_weight)
        del non_quant_model, non_quant_model_modules

    for name, param in model.named_parameters():
        param.requires_grad = False
    merged_model = model.merge_and_unload()
    merged_model.train(False)
    base_model.save_pretrained(output_path, state_dict=merged_model.state_dict())

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if tokenizer_path is None else tokenizer_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
