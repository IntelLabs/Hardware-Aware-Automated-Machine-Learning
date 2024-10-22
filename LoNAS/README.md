# LoNAS

Official implementation of [LoNAS: Elastic Low-Rank Adapters for Efficient Large Language Models]().

This repo contains the code for **LoNAS**, which is a pioneering method that leverages Neural Architecture Search (NAS) to explore a space of elastic low-rank adapters, effectively compressing large language models while maintaining or even enhancing performance, thus facilitating their use in resource-constrained environments. Please refer to our [paper]() for more details.

## Setup

Here is an installation script developed from scratch for **LoNAS**.

```
conda create -n lonas -y python=3.10
conda activate lonas

# install pytorch
pip install torch==2.1.2

# install dependencies
bash install.sh
```
Note: Please ignore the whitespace issues when applying the patch and running `install.sh`.

## Quick Start

### Training

Taking the unified commonsense reasoning training as an example, please download the 15K 
instruction-following [commonsense reasoning training data](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_15k.json) 
from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), and place it under `DATA_PATH`. 

Example command to train a super-adapter of LLaMA-7B using LoNAS:

```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python run_commonsense.py \
    --dataset_path $DATA_PATH//commonsense_15k.json \
    --model_name_or_path yahma/llama-7b-hf \
    --do_train \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 6 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir trained_super_adapter/unified_commonsense/lonas-llama-7b-commonsense \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --target_modules q_proj,k_proj,v_proj,up_proj,gate_proj,down_proj \
    --nncf_config nncf_config/unified_commonsense/nncf_lonas_llama_7b.json
```

The `nncf_config` indicates the NNCF configuration encompassing the search space for elastic adapters and modules of the base model (e.g., `q_proj`). 
The implementation of the elastic modules leverages the BootstrapNAS feature of [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf).
We employ the stage LR scheduler within NNCF, so the learning rate schedule is specified within the NNCF configuration file, 
rather than within the arguments of `TrainingArguments`. For instance, 
```json
"schedule": {
    "list_stage_descriptions": [
        {"train_dims": ["width"], "epochs": 6, "depth_indicator": 1, "width_indicator": 5, "init_lr": 3e-4, "epochs_lr": 6, "sample_rate": 1}
    ]
},
```
For more details on the stage scheduler, see [BootstrapNAS.md](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/nas/bootstrapNAS/BootstrapNAS.md).
After training, the weights of the trained super-adapter will be obtained in the `--output_dir` directory.


### Evaluation

All evaluation datasets can be downloaded from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).
Place them into the directory `datasets/`.
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
mv LLM-Adapters/dataset/ datasets/ 
```

Example command to evaluate the trained super-adapter (heuristic subnetwork):

```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python run_commonsense.py \
    --dataset_path None \
    --model_name_or_path yahma/llama-7b-hf \
    --lora \
    --lora_weights trained_super_adapter/unified_commonsense/lonas-llama-7b-commonsense \
    --nncf_config nncf_config/unified_commonsense/nncf_lonas_llama_7b.json \
    --do_test \
    --output_dir trained_super_adapter/unified_commonsense/lonas-llama-7b-commonsense/results
```

This command evaluates the performance of the heuristic subnetwork across eight commonsense reasoning tasks: 
`BoolQ`, `PIQA`, `SIQA`, `HellaSwag`, `WinoG`, `Arc-e`, `Arc-c`, and `OBQA`.

### Search

In order to discover more optimized subnetworks within the trained super-network, LoNAS employs advanced search algorithms 
to further explore the super-network. To implement it, we leverage [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf), 
which conveniently supports various search algorithms, requiring the configuration of search settings within NNCF config, 
such as:
```json
"search": {
    "algorithm": "NSGA2",
    "batchnorm_adaptation": {
        "num_bn_adaptation_samples": 0
    },
    "num_evals": 200,
    "population": 5,
    "ref_acc": 0.45,
    "acc_delta": 0.01
}
```

Further details can be found in [BootstrapNAS.md](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/nas/bootstrapNAS/BootstrapNAS.md). 
The following is an example command to search for the trained super adapter:

```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python run_commonsense.py \
    --dataset_path $DATA_PATH//commonsense_15k.json \
    --model_name_or_path yahma/llama-7b-hf \
    --lora \
    --lora_weights trained_super_adapter/unified_commonsense/lonas-llama-7b-commonsense \
    --val_set_size 1000
    --nncf_config nncf_config/unified_commonsense/nncf_lonas_llama_7b.json \
    --do_search \
    --output_dir trained_super_adapter/unified_commonsense/lonas-llama-7b-commonsense/search
```

The argument `--val_set_size 1000` signifies the utilization of 1k validation samples to evaluate each discovered 
subnetwork. After running this command, the results of the 200 identified subnetworks (`"num_evals": 200` set in the `search` field of NNCF config) will be placed in the `--output_dir` folder, including `search_progression.png` and `search_progression.csv`. 
From these results, we can select the subnetwork configurations that best meet different requirements.


## Released Models


| Name                                                                                          | Tasks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Base Model                                                                |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **[lonas-bert-base-glue](https://huggingface.co/IntelLabs/lonas-bert-base-glue)**             | [RTE](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-rte), [MRPC](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-mrpc), [STS-B](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-stsb), [CoLA](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-cola), [SST2](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-sst2), [QNLI](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-qnli), [QQP](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-qqp),  [MNLI](https://huggingface.co/IntelLabs/lonas-bert-base-glue/tree/main/lonas-bert-base-mnli) | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| **[lonas-llama-7b-commonsense](https://huggingface.co/IntelLabs/lonas-llama-7b-commonsense-adapter)** | Commonsense Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)             |
| **[lonas-bloomz-7b-math](https://huggingface.co/IntelLabs/lonas-bloomz-7b-math)**             | Math Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | [bigscience/bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)     |

## Reproduce Results

Please refer to `running_commands` for all commands related to reproducing the paper's results.

- GLUE benchmark

| Method      | Trainable Parameter Ratio | GFLOPs     | RTE   | MRPC  | STS-B | CoLA  | SST-2 | QNLI  | QQP   | MNLI  | AVG       |
|-------------|---------------------------|------------|-------|-------|-------|-------|-------|-------|-------|-------|-----------|
| LoRA        | 0.27%                     | 11.2       | 65.85 | 84.46 | 88.73 | 57.58 | 92.06 | 90.62 | 89.41 | 83.00 | 81.46     |
| **LoNAS**   | 0.27%                     | **8.0**    | 70.76 | 88.97 | 88.28 | 61.12 | 93.23 | 91.21 | 88.55 | 82.00 | **83.02** |

- Commonsense Reasoning

| Method      | Total Params.  | TFLOPs    | BoolQ | PIQA | SIQA | HellaSwag | WinoG | Arc-e | Arc-c | OBQA | Average        |
|-------------|----------------|-----------|-------|------|------|-----------|-------|-------|-------|------|----------------|
| LoRA        | 6.7B           | 1.7       | 62.6  | 75.3 | 67.9 | 52.9      | 58.6  | 79.2  | 58.3  | 71.2 | **65.8**       |
| **LoNAS**   | **5.6B**       | **1.4**   | 62.9  | 73.0 | 68.7 | 51.4      | 63.9  | 72.3  | 58.5  | 71.0 | 65.2           |

- Math Reasoning

| Method     | Total Params. | TFLOPs    | GSM8K | AQuA | MAWPS | SVAMP | Average   |
|------------|---------------|-----------|-------|------|-------|-------|-----------|
| LoRA       | 7.1B          | 1.8       | 17.4  | 21.3 | 70.2  | 41.0  | **37.5**  |
| **LoNAS**  | **6.1B**      | **1.5**   | 18.6  | 22.0 | 76.5  | 31.8  | 37.2      |


# Citation

```bibtex
@inproceedings{
munoz2024lonas,
title={LoNAS: Elastic Low-Rank Adapters for Efficient Large Language Models},
author={J. Pablo Muñoz and Jinjie Yuan and Yi Zheng and Nilesh Jain},
booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
year={2024},
url={https://aclanthology.org/2024.lrec-main.940/}
}
```


## Acknowledgement
This work benefits from the following repositories:

- Peft: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- LLM-Adapters: [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- NNCF: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- BootstrapNAS: [https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS)
