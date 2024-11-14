# :scissors:Shears: Fine-tuning For Efficient Models

Official implementation of [Shears: Unstructured Sparsity with Neural Low-rank Adapter Search](https://arxiv.org/abs/2404.10934). :fire:

This repo contains the code for **Shears**, a practical and novel solution that generates efficient models fine-tuned for downstream-specific tasks for real-world applications. Please refer to our paper for more details.

## News
- **[2024.04.18]**  **Shears V1** paper has been released ([link](https://arxiv.org/abs/2404.10934)) and **accepted by NAACL 2024 (Industry Track)**. :books:
- **[2024.04.11]**  Release training and inference code for **Shears V1**. :tada:

## Released Models 🤗

We have released several models fine-tuned with Shears. Find them in the Table below:

| Name                                                                                                                  | Super-network                                                                                                     | Sparsity | Train Data                                                                                                               |
|-----------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------|
| [shears-llama-7b-50-math-heuristic-adapter](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-heuristic-adapter)               | [shears-llama-7b-50-math-super-adapter](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-super-adapter)   | 50%      | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)                  |
| [shears-llama-7b-50-cs-heuristic-adapter](https://huggingface.co/IntelLabs/shears-llama-7b-50-cs-heuristic-adapter) | [shears-llama-7b-50-cs-super-adapter](https://huggingface.co/IntelLabs/shears-llama-7b-50-cs-super-adapter)       | 50%      | [Unified Commonsense](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json)   |
| [shears-llama-13b-50-math-heuristic-adapter](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-heuristic-adapter)             | [shears-llama-13b-50-math-super-adapter](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-super-adapter) | 50%      | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)                  |
| [shears-mpt-7b-50-gsm8k-heuristic-adapter](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-heuristic-adapter)                 | [shears-mpt-7b-50-gsm8k-super-adapter](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-super-adapter)     | 50%      | [GSM8K](https://huggingface.co/datasets/gsm8k)                                                                           |

## Overview

Efficiency Comparison (use LLaMA-13B as an example):

|                 Method                    | Sparsity | Non-zero Parameters | Accuracy |
|:-----------------------------------------:|:--------:|:-------------------:|:--------:|
| [LoRA](https://github.com/microsoft/LoRA) |    -     |        13.0B        |   51.1   |
|                  Shears                   |   50%    |        6.7B         |   50.9   |

By incorporating elastic LoRA adapters into a sparsified base model, Shears can fine-tune a language model without sacrificing the sparsity obtained from the original model weights. This produces sparse models with improvements or minor drops in accuracy and a fraction of the cost compared to other approaches. The increase in sparsity can result in a significant speedup when using runtimes that take advantage of these patterns. 

Overall, Shears has a well-designed, simple yet effective, powerful, and general pipeline that allows users to easily extend it to their desired scenarios/tasks, even audio and video. Feel free to try Shears for any downstream task with any model. :relaxed:

## Setup

Here is an installation script developed from scratch for **Shears**.

```
conda create -n shearsv1 -y python=3.10
conda activate shearsv1

# install pytorch
pip install torch==2.1.2

# install dependencies
bash install.sh
```
Note: Please ignore the whitespace issues when applying the patch and running `install.sh`.

## Quick Start

### Inference

The following code shows an example of loading our trained Shears model: 
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("IntelLabs/shears-mpt-7b-50-base")
model = PeftModel.from_pretrained(base_model, "IntelLabs/shears-mpt-7b-50-gsm8k-heuristic-adapter")
```
Below is an example of generating the instruction-following responses for some math reasoning samples:
```bash
python example_math.py --base_model_path IntelLabs/shears-mpt-7b-50-base --adapter_model_path IntelLabs/shears-mpt-7b-50-gsm8k-heuristic-adapter
```

### Training

#### Step 1. Unstructured Sparsifying

Before fine-tuning, Shears employs a simple but effective pruning approach [Wanda](https://arxiv.org/abs/2306.11695) to sparsify the language model, serving as the base model (frozen) for adapter training.
Clone the [Wanda](https://github.com/locuslab/wanda) repo:

```bash
git clone https://github.com/locuslab/wanda.git && cd wanda && git checkout 8e8fc87 && cd ..
```

Below is an example command for unstructured sparsifying LLaMA-7B with Wanda to achieve unstructured 50% sparsity (takes about five minutes).
```bash
SPARSE_MODEL_PATH=shears-llama-7b-50-base

python wanda/main.py \
    --model yahma/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save wanda_out \
    --save_model $SPARSE_MODEL_PATH
```
- `--model`: The identifier for the model on the Hugging Face model hub or local path.
- `--sparsity_ratio`: Specifies the percentage of weights to be pruned.
- `--save_model`: Specifies the directory where the pruned language model will be stored.

Further details can be referred to [Wanda](https://github.com/locuslab/wanda). You can also skip this step and adopt our released sparsified models 
(find them in Base Model of [Table](#released-models-)). 
It is worth noting that the sparsifying step can be replaced by any other sparse 
(or even quantization) algorithm. Feel free to try other approaches for the base model.


#### Step 2. Adapter Training

Taking the unified math reasoning training as an example, please download the 10K instruction-following math reasoning training data ([link](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)) from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), and place it under `DATA_PATH`. 

Example command to train the super-adapter of the pruned LLaMA-7B using Shears:

```bash
ADAPTER_MODEL_PATH=trained_super_adapter/shears-llama-7b-50-math-super-adapter
NNCF_CONFIG=nncf_config/nncf_shears_llama.json

python run_math.py \
    --dataset_path $DATA_PATH/math_10k.json \
    --model_name_or_path $SPARSE_MODEL_PATH \
    --do_train \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir $ADAPTER_MODEL_PATH \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
    --nncf_config $NNCF_CONFIG
```

`--nncf_config` indicates the NNCF configuration, including the search space for elastic adapters.
To implement the elastic adapter, we apply the [BootstrapNAS](https://github.com/openvinotoolkit/nncf/tree/develop/nncf/experimental/torch/nas/bootstrapNAS) feature supported in [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf), which provides a suite of compression algorithms for neural network optimization.
The NNCF configuration details are in [nncf_config.md](./nncf_config/nncf_config.md).
After training, the trained super-adapter will be saved in `ADAPTER_MODEL_PATH`.

### Evaluation

All evaluation datasets can be downloaded from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).
Place them into the directory `datasets/`.
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
mv LLM-Adapters/dataset/ datasets/ 
```

Example command to evaluate the trained super-adapter (heuristic subnetwork):

```bash
python run_math.py \
    --model_name_or_path $SPARSE_MODEL_PATH \
    --lora \
    --lora_weights $ADAPTER_MODEL_PATH \
    --do_test \
    --output_dir $ADAPTER_MODEL_PATH/results \
    --nncf_config $NNCF_CONFIG
```

The above command can also be used to test the released model, for example,
```bash
python run_math.py \
    --model_name_or_path $SPARSE_MODEL_PATH \
    --lora \
    --lora_weights IntelLabs/shears-llama-7b-50-math-super-adapter \
    --do_test \
    --output_dir ./results \
    --nncf_config nncf_config/nncf_shears_llama.json
```
Note that the torch version we used in our experiments is `1.12.1+cu113`, and the results might vary with different versions.

## Reproduce results

Please refer to [running_commands](./running_commands) for all commands related to reproducing the paper's results.

- LLaMA with Math Reasoning tasks

| Model                 | Sparsity    | GSM8K | AQuA  | MAWPS | SVAMP | Average |
|-----------------------|-------------|-------|-------|-------|-------|---------|
| LLaMA-7B-LoRA         | -           | 37.5  | 18.9  | 79.0  | 52.1  | 46.9    |
| **LLaMA-7B-Shears**   | **40%**     | 36.8  | 19.7  | 83.2  | 47.7  | 46.9    |
| **LLaMA-7B-Shears**   | **50%**     | 36.1  | 22.0  | 78.6  | 44.5  | 45.3    |
| LLaMA-13B-LoRA        | -           | 47.5  | 18.5  | 83.6  | 54.6  | 51.1    |
| **LLaMA-13B-Shears**  | **40%**     | 48.3  | 21.3  | 83.2  | 55.2  | 52.0    |
| **LLaMA-13B-Shears**  | **50%**     | 45.1  | 22.0  | 83.2  | 53.3  | 50.9    |

- LLaMA with Commonsense Reasoning tasks

| Model                | Sparsity | BoolQ    | PIQA  | SIQA  | HellaSwag | WinoG  | ARC-e   | ARC-c   | OBQA   | Average |
|----------------------|----------|----------|-------|-------|-----------|--------|---------|---------|--------|---------|
| ChatGPT              | -        | 73.1     | 85.4  | 68.5  | 78.5      | 66.1   | 89.8    | 79.9    | 74.8   | 77.0    |
| LLaMA-7B-LoRA        | -        | 68.9     | 80.7  | 77.4  | 78.1      | 78.8   | 77.8    | 61.3    | 74.8   | 74.7    |
| **LLaMA-7B-Shears**	 | **40%**  | 67.0     | 79.9  | 76.7  | 80.1      | 78.6   | 76.9    | 62.3    | 77.8   | 74.9    |
| **LLaMA-7B-Shears**	 | **50%**  | 67.3     | 79.1  | 77.5  | 73.3      | 77.7   | 74.4    | 57.9    | 72.8   | 72.5    |

- MPT with GSM8K

| Sparsity | 0%    | 40%  | 50%  | 60%  | 70%  |
|----------|-------|------|------|------|------|
| Accuracy | 36.1  | 35.7 | 33.4 | 30.4 | 22.8 |


## Explore a trained super-adapter

To enhance exploration of the super-network trained using the Shears method, we provide an illustrative example `search/load_and_explore_supernet.ipynb`. This notebook demonstrates the direct loading of a Shears super-network and the extraction of various subnetworks. 
This facilitates users in applying their own search algorithms and evaluation metrics to extract subnetworks tailored to their specific requirements.

## Extract and save a sub-adapter

After training, we obtain the weights of the super-adapter stored in `ADAPTER_MODEL_PATH` and activate different 
sub-networks using NNCF. However, once a particular sub-network we need is identified, activating just that one sub-network using NNCF is no longer necessary. Instead, what we need is a clean, pruned, and directly-loadable 
sub-network. For this purpose, we provide a function to extract/save any sub-adapter (please refer to the end of 
`search/load_and_explore_supernet.ipynb`). Below is an example to obtain the heuristic sub-adapter of a trained 
super-adapter:

```python
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM
from search.supernet import ShearsSuperNet
from utils.utils import load_nncf_config

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
nncf_config = load_nncf_config(NNCF_CONFIG, num_hidden_layers=model.config.num_hidden_layers)

supernet = ShearsSuperNet.from_checkpoint(model, nncf_config, supernet_elasticity_path=None, supernet_weights_path=None)
supernet.activate_heuristic_subnet()
supernet.extract_and_save_active_sub_adapter(super_adapter_dir=ADAPTER_MODEL_PATH, sub_adapter_dir=os.path.join(ADAPTER_MODEL_PATH, "heuristic_adapter"))
```

We released some examples of the extracted heuristic sub-adapters. Refer to them in [Table](#released-models-).

## Citation
If you find our Shears code and paper helpful, please kindly cite:
```bibtex
@inproceedings{munoz-etal-2024-shears,
    title = "Shears: Unstructured Sparsity with Neural Low-rank Adapter Search",
    author = "Mu{\~n}oz, J. Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",
    editor = "Yang, Yi  and
      Davani, Aida  and
      Sil, Avi  and
      Kumar, Anoop",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-industry.34",
    doi = "10.18653/v1/2024.naacl-industry.34",
    pages = "395--405",
}
```

## Acknowledgement
This work benefits from the following repositories:

- LLaMA: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
- MPT: [https://www.mosaicml.com/mpt](https://www.mosaicml.com/mpt)
- Transformers: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- PEFT: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- LLM-Adapters: [ttps://github.com/AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- NNCF: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- BootstrapNAS: [https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS)
- Wanda: [https://github.com/locuslab/wanda](https://github.com/locuslab/wanda)
