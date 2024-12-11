# MultiPruner

Official implementation of [Fine-Grained Training-Free Structure Removal in Foundation Models]().

This repo contains the code for **MultiPruner**, a novel pruning approach that surpasses recent training-free pruning 
methods by adopting a multidimensional, iterative, fine-grained pruning strategy.
Please refer to our paper for more details.

## News
- **[2025.xx.xx]** Release the code for **MultiPruner**. :tada:

## Setup

Here is an installation script developed from scratch.

```
pip install virtualenv
virtualenv multipruner-env
source multipruner-env/bin/activate
pip install torch==2.3.1

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

We have provided some running commands (including pruning and recovery-tuning) and pruning results of MultiPruner, which can be found [here](./results).

In addition to the 22% pruning ratio shown in the paper, we also explored pruning ratios that result in 1%, 2%, and 3% 
accuracy degradation (compared to Dense), under both `without finetune` and `with finetune` scenarios. 
This investigation may facilitate practical applications. The results of Llama-2-7b-hf are shown in the following table:

| Method                   | Pruning Ratio | Acc. (%) | Acc. Drop | Relative Acc. |
|--------------------------|---------------|----------|-----------|---------------|
| Dense                    | /             | 68.96    | /         | 100%          |
| MultiPruner w/o finetune | 7%            | 67.94    | -1.02%    | 98.52%        |
| MultiPruner w/o finetune | 10%           | 67.02    | -1.94%    | 97.19%        |
| MultiPruner w/o finetune | 14%           | 65.93    | -3.03%    | 95.61%        |
| MultiPruner w/ finetune  | 12%           | 68.28    | -0.68%    | 99.01%        |
| MultiPruner w/ finetune  | 15%           | 67.41    | -1.55%    | 97.75%        |
| MultiPruner w/ finetune  | 18%           | 66.16    | -2.80%    | 95.94%        |


## Released Pruned Models 🤗

We have released several compressed models by MultiPruner:

| Source Model                                                                            | Pruning Ratio | Recovery Tuning | Pruned Model                                                                                                  |
|-----------------------------------------------------------------------------------------|---------------|-----------------|---------------------------------------------------------------------------------------------------------------|
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 7%            | ✘               | [IntelLabs/MultiPruner-Llama-2-6.3b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-6.3b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 10%           | ✘               | [IntelLabs/MultiPruner-Llama-2-6.1b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-6.1b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 12%           | ✘               | [IntelLabs/MultiPruner-Llama-2-5.9b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.9b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 12%           | ✔               | [IntelLabs/MultiPruner-Llama-2-5.9b-alpaca](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.9b-alpaca) |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 14%           | ✘               | [IntelLabs/MultiPruner-Llama-2-5.8b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.8b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 15%           | ✘               | [IntelLabs/MultiPruner-Llama-2-5.7b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.7b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 15%           | ✔               | [IntelLabs/MultiPruner-Llama-2-5.7b-alpaca](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.7b-alpaca) |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 18%           | ✘               | [IntelLabs/MultiPruner-Llama-2-5.5b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.5b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 18%           | ✔               | [IntelLabs/MultiPruner-Llama-2-5.5b-alpaca](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.5b-alpaca) |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 22%           | ✘               | [IntelLabs/MultiPruner-Llama-2-5.3b](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.3b)               |
| [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)             | 22%           | ✔               | [IntelLabs/MultiPruner-Llama-2-5.3b-alpaca](https://huggingface.co/IntelLabs/MultiPruner-Llama-2-5.3b-alpaca) |
| [Qwen/Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)                               | 22%           | ✘               | [IntelLabs/MultiPruner-Qwen1.5-6b](https://huggingface.co/IntelLabs/MultiPruner-Qwen1.5-6b)                   |
| [baichuan-inc/Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base) | 22%           | ✘               | [IntelLabs/MultiPruner-Baichuan2-5.8b](https://huggingface.co/IntelLabs/MultiPruner-Baichuan2-5.8b)           |

### Loading the compressed model for evaluation

```bash
python eval.py --model_path <path to compressed model> --output_path <path to evaluation results>
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
