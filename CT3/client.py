import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

try:
    from ipex_llm.transformers import AutoModelForCausalLM as IpexAutoModelForCausalLM
except ImportError:
    print("ipex-llm is not installed. Please install ipex-llm to use this feature.")


from ct3_config import add_ct3_args, parse_ct3_args
from models.model import convert_to_ttt_model


URL = "http://127.0.0.1:8000/search/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model."
    )
    add_ct3_args(parser)
    args = parser.parse_args()
    model_path = args.model_path
    set_seed(42)
    ct3_config = parse_ct3_args(args)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if ct3_config.use_ipex_llm:
        model = IpexAutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_low_bit="bf16",
            optimize_model=False,
            torch_dtype=torch.bfloat16,
            # modules_to_not_convert=["lm_head"],
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        ).eval()

    model = convert_to_ttt_model(model, tokenizer, ct3_config, ct3_server_url=URL)
    
    while True:
        query = input("Please enter your question (type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Thank you for using the service. Goodbye!")
            break

        input_ids = tokenizer(query, return_tensors="pt").to("cuda")
        output = model.generate(**input_ids, max_new_tokens=256, do_sample=False)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(output)


if __name__ == "__main__":
    main()
