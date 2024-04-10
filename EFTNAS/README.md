# EFTNAS

Official implementation of [Searching for Efficient Language Models in First-Order Weight-Reordered Super-Networks]() :boom:.

This repo contains the code for a practical and novel solution called **EFTNAS** that combines neural architecture search and network pruning to efficiently compress transformer-based models for various resource budgets, achieving significant size reductions of over 5x without compromising performance on NLP tasks. Please refer to our [paper]() for more details.

## Overview

<p align="center">
<img src="./figures/eftnas_pipeline.png" alt="EFTNAS Pipeline" width="600"/>
</p>

Unstructured weight importance information is used to obtain
subnetwork configurations at the boundaries of the desired search space. Then, intermediate subnetworks
are identified depending on the required complexity for the search space, and all subnetwork configurations
are combined, effectively automating the search space generation based on the desired computation
complexity. Weights are then reordered, elasticity is enabled, and the super-network is trained. Finally,
high-performing Transformer-based subnetworks are discovered for a variety of performance targets.

## Setup

Here is an installation script developed from scratch for EFTNAS.

```
conda create -n eftnas -y python=3.10
conda activate eftnas

# install pytorch
pip install torch==2.1.2

# install dependencies
bash install.sh
```

## Quick Start

EFTNAS consists of two stages: i) **Search Space Generation** and ii) **Training**.

### Step 1. Search Space Generation

#### Train importance weight via movement sparsity

Firstly, to obtain the weight importance for both weight re-ordering and search space generation, EFTNAS employs 
an effective learning-based unstructured sparsification algorithm [Movement Pruning (Sanh et al., 2020)](https://arxiv.org/pdf/2005.07683.pdf).
This method can more accurately generate weight importance scores for specific tasks.

Below is an example command for movement sparsity training to obtain the weight importance:

```bash
EFTNAS_PATH=$PWD
cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name cola \
    --nncf_config ${EFTNAS_PATH}/eftnas_configs/nncf_base_config_for_bert_base.json \
    --output_dir ${EFTNAS_PATH}/trained_models/eftnas-bert-base-cola/movement_sparsity \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --only_generate_importance_weight True
```

- `--model_name_or_path`: the pre-trained model we use as the backbone model.
- `--task_name`: the task of GLUE benchmark.
- `--nncf_config`: the NNCF configuration including the config of movement sparsity
(refer to [MovementSparsity.md](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/sparsity/movement/MovementSparsity.md)
for more details).
- `--output_dir`: the directory to save importance weights.

After movement sparsity training, the importance for the pretrained weights of `--model_name_or_path` can 
be obtained in the `--output_dir` directory, which will be utilized for search space generation and weight-reorder during 
NAS training.

#### Generate search space

Based on the trained weight importance, EFTNAS has a well-designed algorithm to automatically generate 
the search space for the super-network.
Below is an example command for search space generation using the weight importance:
```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python eftnas_search_space/generate_eftnas_search_space.py \
  --source_config eftnas_configs/nncf_base_config_for_bert_base.json \
  --model_name_or_path bert-base-uncased \
  --importance_weight_dir trained_models/eftnas-bert-base-cola/movement_sparsity \
  --target_config generated_configs/nncf_eftnas_s1_bert_base_cola.json
```
- `--source_config`: NNCF config before search space generation.
- `--importance_weight_dir`: directory to save importance weights obtained from the previous step.
- `--target_config`: NNCF config after search space generation.

The generated search space will be saved in `--target_config`.

### Step 2. Training

Once the weight importance scores and the NNCF configuration with the automatically generated search space are obtained, 
EFTNAS conducts NAS training utilizing based on the information from Step 1. 

Due to the feature of BootstrapNAS, we need to manually set the epoch and learning rate for NAS training in the 
NNCF configuration (please refer to [BootstrapNAS.md](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/nas/bootstrapNAS/BootstrapNAS.md)
for more details), e.g.,
```json
"schedule": {
    "list_stage_descriptions": [
        {"train_dims": ["width"], "epochs": 12, "depth_indicator": 1, "width_indicator": 4, "init_lr": 2e-5, "epochs_lr": 12, "sample_rate": 1}
    ]
}
```

Below is an example command of training:

```bash
EFTNAS_PATH=$PWD
cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --kd_teacher_model ModelTC/bert-base-uncased-cola \
    --reorg_cache_model ${EFTNAS_PATH}/trained_models/eftnas-bert-base-cola/movement_sparsity/pytorch_model.bin \
    --task_name cola \
    --nncf_config ${EFTNAS_PATH}/generated_configs/nncf_eftnas_s1_bert_base_cola.json \
    --output_dir ${EFTNAS_PATH}/trained_models/eftnas-bert-base-cola \
    --do_train \
    --do_eval \
    --do_search \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16
```

- `--kd_teacher_model`: the pre-trained model we use as the teacher model in knowledge distillation.
- `--reorg_cache_model`: directory to save importance weights obtained from the movement sparsity step.
- `--nncf_config`: the NNCF configuration generated in the search space generation step.


## Released Models

- Trained super-network and weight importance

| Name                        | Tasks                                                                            | Base Model          |
|-----------------------------|----------------------------------------------------------------------------------|---------------------|
| **[eftnas-s1-bert-base]()** | [CoLA](), [MNLI](), [MRPC](), [QNLI](), [QQP](), [RTE](), [SST2](), [SQuADv1.1](), [SQuADv2.0]() | [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) |
| **[eftnas-s2-bert-medium]()** | [CoLA](), [MNLI](), [MRPC](), [QNLI](), [QQP](), [RTE](), [SST2](), [SQuADv1.1](), [SQuADv2.0]() | [google/bert_uncased_L-8_H-512_A-8](https://huggingface.co/google/bert_uncased_L-8_H-512_A-8) |

- Generated search space 

The search space generated by our approach ([Step 1. Search Space Generation](#step-1-search-space-generation)) can be found in the NNCF 
configuration of the `eftnas_configs/` directory.

## Reproduce Results

Please refer to `running_commands` for all commands related to reproducing the paper's results.

| Model                         | GFLOPs    | GLUE Avg.     | MNLI-m   | QNLI | QQP      | SST-2    | CoLA     | MRPC     | RTE  |
|-------------------------------|-----------|---------------|----------|------|----------|----------|----------|----------|------|
| **Development Set**           |
| BERT<sub>base</sub> (teacher) | 11.2      | 83.3          | 84.7     | 91.8 | 91.0     | 93.2     | 59.6     | 90.4     | 72.5 |
| DistilBERT<sub>6</sub>        | 5.7       | 78.6          | 82.2     | 89.2 | 88.5     | 91.3     | 51.3     | 87.5     | 59.9 |
| TinyBERT<sub>6</sub>          | 5.7       | 81.9          | 84.5     | 91.1 | 91.1     | 93.0     | 54.0     | 90.6     | 73.4 |
| MiniLM                        | 5.7       | 81.0          | 84.0     | 91.0 | 91.0     | 92.0     | 49.2     | 88.4     | 71.5 |
| MiniLMv2 (6x768)              | 5.7       | 81.7          | 84.2     | 90.8 | 91.1     | 92.4     | 52.5     | 88.9     | 72.1 |
| **EFTNAS-S1 (Ours)**          | 5.7       | **82.9**      | **84.6** | 90.8 | **91.2** | **93.5** | **60.6** | **90.8** | 69.0 |
| NAS-BERT<sub>10</sub> + KD    | 2.3       | 74.2          | 76.4     | 86.3 | 88.5     | 88.6     | 34.0     | 79.1     | 66.6 |
| AutoDistil<sub>Proxy-S</sub>  | 2.0       | 79.9          | 83.2     | 90.0 | 90.6     | 90.1     | 48.3     | 88.3     | 69.4 |
| AutoDistil<sub>Agnostic</sub> | 2.1       | 79.6          | 82.8     | 89.9 | 90.8     | 90.6     | 47.1     | 87.3     | 69.0 |
| **Test Set**                  |
| BERT<sub>base</sub> (teacher) | 11.2      | 78.2          | 84.6     | 90.5 | 71.2     | 93.5     | 52.1     | 88.9     | 66.4 |
| DistilBERT<sub>6</sub>        | 5.7       | 76.8          | 82.6     | 88.9 | 70.1     | 92.5     | 49.0     | 86.9     | 58.4 |
| TinyBERT<sub>6</sub>          | 5.7       | 79.4          | 84.6     | 90.4 | 71.6     | 93.1     | 51.1     | 87.3     | 70.0 |
| MiniLMv2 (6x768)              | 5.7       | 77.5          | 83.8     | 90.2 | 70.9     | 92.9     | 46.6     | 89.1     | 69.2 |
| **EFTNAS-S1 (Ours)**          | 5.7       | 77.7          | 83.7     | 89.9 | **71.8** | **93.4** | **52.6** | 87.6     | 65.0 |
| **EFTNAS-S2 (Ours)**           | 2.2       | 75.2          | 82.0     | 87.8 | 70.6     | 91.4     | 44.5     | 86.1     | 64.0 |


## Citation

```bibtex
@inproceedings{
  eftnas2024,
  title={Searching for Efficient Language Models in First-Order Weight-Reordered Super-Networks},
  author={J. Pablo Munoz and Yi Zheng and Nilesh Jain},
  booktitle={The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation},
  year={2024},
  url={}
}
```

## Acknowledgement
This work benefits from the following repositories:

- NNCF: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- BootstrapNAS: [https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS)
