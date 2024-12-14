# MultiPruner

Official implementation of [Fine-Grained Training-Free Structure Removal in Foundation Models]().

This repo contains the code for **MultiPruner**, a novel pruning approach that surpasses recent training-free pruning 
methods, e.g., BlockPruner (Zhong el al., 2024) and ShortGPT (Men et al., 2024), by adopting a multidimensional, iterative, fine-grained pruning strategy.
Please refer to our paper for more details.

## News
- **[2024.12.14]** Release the code for **MultiPruner**. :tada:

## Supported Models 🤗

- **Llama**
  - [x] [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
  - [x] [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  - [x] [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
  - [x] [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
- **Qwen**
  - [x] [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
  - [x] [Qwen/Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)
  - [x] [Qwen/Qwen1.5-14B](https://huggingface.co/Qwen/Qwen1.5-14B)
- **Baichuan**
  - [x] [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
  - [x] [baichuan-inc/Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)

**All pruning result configurations and pruning commands are available [here](./results).**

## Setup

Use the following instructions to create a virtual environment with the required dependencies.

```
# install dependencies
bash install.sh
```

## Run

We use [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model as an example.

### Prune

```bash
python run_multipruner.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio 22.00 \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

- `model_path`: Path to the pre-trained model.
- `output_path`: Directory to save the pruning and evaluation results.
- `weight_reorder`: Indicates that weight reordering should be performed in Attn and MLP.
- `do_prune`: Flag to indicate whether to perform pruning.
- `target_ratio`: Target pruning ratio.
- `pruning_distribution`: Pruning ratio distribution for different granularities.
- `mlp_channel_group_size`: Number of channels for each group (MLP).
- `attn_channel_group_size`: Number of channels for each group (Attn), generally a multiple of the head dimension.
- `importance_metric`: Metric for calculating block importance, currently only supports PPL.
- `calibration_dataset`: Calibration dataset name ("alpaca", "c4", "ptb" or "wikitext2").
- `num_calibration_samples_block`: Number of calibration samples to use for depth (block) pruning (stage 1).
- `num_calibration_samples_width`: Number of calibration samples to use for width pruning (stage 2 and 3).
- `do_eval`: Flag to indicate whether to perform evaluation.

### Extract the Compressed Model

The final compressed model can be extracted based on the optimal pruning configuration obtained from MultiPruner.
For more details, please refer to [this link](./extract).
Below is an example of how to extract a pruned Llama-2-7B:

```bash
python extract/extract_model.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --weight_reorder \
  --pruned_model_config_file <path to pruning results>/pruning_config.json \
  --output_path <path to compressed model>
```

### Recovery Finetuning

After we have obtained the pruned model, we can use the Alpaca dataset for recovery fine-tuning. 
More details can be found [here](./recovery).
The following is an example command for the compressed Llama-2-7B:

```bash
# Finetune the compressed model
python recovery/finetune.py \
  --model_path <path to compressed model> \
  --do_train \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --output_path <path to finetuned compressed model> \
  --do_eval
```

## Results

We have provided some running commands (including pruning and recovery-tuning) and pruning configurations of MultiPruner, which can be found [here](./results).

#### Llama-3.1-8B

| Method           | Pruning Ratio | Acc. (%) | WikiText2 PPL |
|------------------|---------------|----------|---------------|
| Dense            | /             | 73.75    | 6.24          |
| BlockPruner      | 10%           | 66.75    | 10.58         |
| **MultiPruner**  | 10%           | 69.27    | 8.93          |
| BlockPruner      | 20%           | 59.08    | 15.37         |
| **MultiPruner**  | 20%           | 63.07    | 13.86         |

#### Meta-Llama-3-8B

| Method           | Pruning Ratio | Acc. (%) | WikiText2 PPL |
|------------------|---------------|----------|---------------|
| Dense            | /             | 72.73    | 6.14          |
| BlockPruner      | 10%           | 66.46    | 10.88         |
| **MultiPruner**  | 10%           | 69.03    | 8.19          |
| BlockPruner      | 20%           | 57.59    | 22.36         |
| **MultiPruner**  | 20%           | 63.02    | 16.01         |

#### Qwen2.5-7B

| Method           | Pruning Ratio | Acc. (%) | WikiText2 PPL |
|------------------|---------------|----------|---------------|
| Dense            | /             | 72.04    | 6.85          |
| BlockPruner      | 10%           | 67.44    | 9.88          |
| **MultiPruner**  | 10%           | 69.71    | 9.15          |
| BlockPruner      | 20%           | 57.44    | 17.17         |
| **MultiPruner**  | 20%           | 62.82    | 13.37         |


For additional results and discussions on other models, please refer to the paper.

In addition, we also explored pruning ratios that result in 1%, 2%, and 3% 
accuracy degradation (compared to Dense), under both `without finetune` and `with finetune` scenarios. 
This investigation may facilitate practical applications. The results of Llama-2-7B are shown in the following table:

| Method                   | Pruning Ratio | Acc. (%) | Acc. Drop | Relative Acc. |
|--------------------------|---------------|----------|-----------|---------------|
| Dense                    | /             | 68.96    | /         | 100%          |
| MultiPruner w/o finetune | 7%            | 67.94    | -1.02%    | 98.52%        |
| MultiPruner w/o finetune | 10%           | 67.02    | -1.94%    | 97.19%        |
| MultiPruner w/o finetune | 14%           | 65.93    | -3.03%    | 95.61%        |
| MultiPruner w/ finetune  | 12%           | 68.28    | -0.68%    | 99.01%        |
| MultiPruner w/ finetune  | 15%           | 67.41    | -1.55%    | 97.75%        |
| MultiPruner w/ finetune  | 18%           | 66.16    | -2.80%    | 95.94%        |

*In all tables, `Acc.(%)` represents the average accuracy score across the five tasks: `piqa`, `winogrande`, `hellaswag`, `arc_easy`, and `arc_challenge`.*

### Loading the compressed model for evaluation

```bash
python eval.py --model_path <path to compressed model> --output_path <path to evaluation results>
```

## Acknowledgement

MultiPruner benefits from the following work:

```bibtex
@article{zhong2024blockpruner,
  title={BlockPruner: Fine-grained Pruning for Large Language Models},
  author={Zhong, Longguang and Wan, Fanqi and Chen, Ruijun and Quan, Xiaojun and Li, Liangzhi},
  journal={arXiv preprint arXiv:2406.10594},
  year={2024}
}
```

## Citation
If you find MultiPruner's code and paper helpful, please kindly cite:
```bibtex
@article{munoz2025multipruner,
  title = {Fine-Grained Training-Free Structure Removal in Foundation Models},
  author = {J. Pablo Munoz and Jinjie Yuan and Nilesh Jain},
  year = {2025},
  url = {}
}
```
