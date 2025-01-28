# Mamba-Shedder

Official implementation of [Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models]().

This repo contains the code for Mamba-Shedder, which explores the compression of the new Mamba-series architectures (and their hybrids). 
We study the sensitivity of these models to the removal of selected components at different granularities to reduce model size and computational overhead, thereby improving their efficiency while maintaining accuracy.
Please refer to our paper for more details.

## News
- **[2025.01.23]** Support for the new hybrid architecture model **Hymba**, please refer to [Hymba-Pruning](./hybrid/Hymba-Pruning).
- **[2025.01.23]** Support Zamba2 ([Zamba2-Pruning](./hybrid/Zamba2-Pruning)).
- **[2025.01.22]** Release the code for **Mamba-Shedder**. :tada:

## Released Pruned Models 🤗

Compressed models by Mamba-Shedder:

| Source Model                                                       | Components Removed | Recovery Tuning | Relative Acc. | Pruned Model Link                                                                      | Inference Speedup |
|--------------------------------------------------------------------|--------------------|-----------------|---------------|----------------------------------------------------------------------------------------|-------------------|
| [Hymba-1.5B-Base](https://huggingface.co/nvidia/Hymba-1.5B-Base)   | 7 Hymba Blocks     | ✘               | 97%           | [Link]()                 | ~1.2x             |
| [Hymba-1.5B-Base](https://huggingface.co/nvidia/Hymba-1.5B-Base)   | 7 Hymba Blocks     | ✔               | 99%           | [Link]()          | ~1.2x             |
| [mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b)       | 14 Mamba Blocks    | ✘               | 90%           | [Link]()                      | ~1.3x             |
| [mamba2-2.7b](https://huggingface.co/state-spaces/mamba2-2.7b)     | 22 SSMs            | ✘               | 96%           | [Link]()        | ~1.2x             |
| [mamba2-2.7b](https://huggingface.co/state-spaces/mamba2-2.7b)     | 22 SSMs            | ✔               | 99%           | [Link]() | ~1.2x             |

<sup>*</sup> *Mamba-Shedder pruned models are currently under internal review and will be released soon.*

## Setup

Use the following instructions to create a virtual environment with the required dependencies.

```
# install dependencies
bash install.sh
```

## Run

### Evaluation before Pruning

```bash
python eval.py --model_path <path to mamba model>
```

### Prune

#### Mamba Block Pruning

An example command for [mamba-2.8b](https://huggingface.co/state-spaces/mamba-2.8b) with Mamba Block Pruning:

```bash
python prune.py \
  --model_path state-spaces/mamba-2.8b \
  --do_prune \
  --output_path <path to pruning results> \
  --prune_target mamba_block \
  --target_pruning_steps 10 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
  --do_eval
```

- `model_path`: Path to the pre-trained Mamba model.
- `do_prune`: Flag to indicate whether to perform pruning.
- `output_path`: Directory to save the pruning and evaluation results.
- `prune_target`: "mamba_block" or "ssm".
- `target_pruning_steps`: Number of pruning target modules (mamba blocks or SSMs).
- `importance_metric`: Metric for calculating block importance, currently only supports PPL.
- `calibration_dataset`: Calibration dataset name ("alpaca", "c4", "ptb" or "wikitext2").
- `num_calibration_samples`: Number of calibration samples for pruning.
- `do_eval`: Flag to indicate whether to perform evaluation.

#### SSM Pruning

An example command for [mamba2-2.7b](https://huggingface.co/state-spaces/mamba2-2.7b) with SSM Pruning:

```bash
python prune.py \
  --model_path state-spaces/mamba2-2.7b \
  --do_prune \
  --output_path <path to pruning results> \
  --prune_target ssm \
  --target_pruning_steps 20 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
  --do_eval
```

### Extract the Pruned Model

Extract the pruned model based on the optimal pruning configuration obtained from Mamba-Shedder. 
For more details, please refer to [here](./extract). 
Here is an example to extract a pruned [mamba2-2.7b](https://huggingface.co/state-spaces/mamba2-2.7b):

```bash
python extract/extract_mamba.py \
  --model_path state-spaces/mamba2-2.7b \
  --pruned_model_config_file <path to pruning results>/pruning_config.json \
  --output_path <path to compressed model>
```

### Recovery Fine-tuning

After we have obtained the pruned model, we can use [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset for recovery fine-tuning:

```bash
# Finetune the compressed Mamba-2
python recovery/finetune_mamba.py \
    --model_path <path to compressed model> \
    --do_train \
    --batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-5 \
    --output_path <path to trained model> \
    --do_eval
```

## Results

All run commands and pruning results can be found in [here](./results).

### Loading the compressed model for evaluation

```bash
python eval.py --model_path <path to compressed model>
```

## Citation
If you find Mamba-Shedder's code and paper helpful, please kindly cite:
```bibtex
@inproceedings{munoz2025mambashedder,
  title = {Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models},
  author = {Mu{\~n}oz, J. Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh},
  booktitle = "Proceedings of the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025)",
    month = jun,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "",
}
```
