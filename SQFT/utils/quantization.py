import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

def main():
    parser = argparse.ArgumentParser(description="Quantize and save a model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer. Defaults to base model path if not provided.")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for model weights.")
    parser.add_argument("--block_name_to_quantize", type=str, default=None, help="Specific block name to quantize.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the quantized model and tokenizer.")
    args = parser.parse_args()

    base_model_path = args.base_model_path
    tokenizer_path = args.tokenizer_path
    dtype = args.dtype
    block_name_to_quantize = args.block_name_to_quantize
    output_dir = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if tokenizer_path is None else tokenizer_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    if block_name_to_quantize is None:
        quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False)
    else:
        quantization_config = GPTQConfig(
            bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False, block_name_to_quantize=block_name_to_quantize
        )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    quantized_model.config.quantization_config.use_exllama = False
    if block_name_to_quantize is not None:
        quantized_model.config.quantization_config.block_name_to_quantize = block_name_to_quantize

    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Uncomment the following lines to push the model and tokenizer to the hub
    # quantized_model.push_to_hub(output_dir, private=True)
    # tokenizer.push_to_hub(output_dir, private=True)

if __name__ == "__main__":
    main()
