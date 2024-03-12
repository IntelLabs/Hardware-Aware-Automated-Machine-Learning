import argparse
import os

import torch

Q_PROJ_RANGE = (0, 4096)
K_PROJ_RANGE = (4096, 8192)
V_PROJ_RANGE = (8192, 12288)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_name_or_path", default="mpt-7b", type=str, help="Path to mpt-7b model")

    args = parser.parse_args()
    base_model_name_or_path = args.base_model_name_or_path
    paths = [
        f"{base_model_name_or_path}/pytorch_model-00001-of-00002.bin",
        f"{base_model_name_or_path}/pytorch_model-00002-of-00002.bin",
    ]
    new_state_dict = {}

    for path in paths:
        old_state_dict = torch.load(path)
        keys = list(old_state_dict.keys())
        for key in keys:
            if key.endswith("Wqkv.weight"):
                prefix = ".".join(key.split(".")[:-2])
                new_state_dict[prefix + ".q_proj.weight"] = old_state_dict[key][
                    Q_PROJ_RANGE[0] : Q_PROJ_RANGE[1]
                ].clone()
                new_state_dict[prefix + ".k_proj.weight"] = old_state_dict[key][
                    K_PROJ_RANGE[0] : K_PROJ_RANGE[1]
                ].clone()
                new_state_dict[prefix + ".v_proj.weight"] = old_state_dict[key][
                    V_PROJ_RANGE[0] : V_PROJ_RANGE[1]
                ].clone()
            else:
                new_state_dict[key] = old_state_dict[key].clone()

    os.system(f"rm {base_model_name_or_path}/pytorch_model*")
    torch.save(new_state_dict, f"{base_model_name_or_path}/pytorch_model.bin")


if __name__ == "__main__":
    main()
