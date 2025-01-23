import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--adapter_model_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    base_model_path = args.base_model_path
    adapter_model_path = args.adapter_model_path
    output_path = args.output_path

    base_model, loading_info = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype="float16",
        output_loading_info=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_model_path, device_map={"": 0})
    model.eval()
    merged_model = model.merge_and_unload()
    merged_model.train(False)

    sd = merged_model.state_dict()
    base_model.save_pretrained(output_path, state_dict=sd)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
