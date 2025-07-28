# CT3+: Boosting Performance on Vision Downstream Tasks through Test-time Training

## Overview

CT3+ is an advanced framework designed to enhance the performance of **Image-Text-to-Text** tasks through Test-Time Training (TTT).
This repository provides the necessary code and instructions to set up and utilize CT3+ effectively.

## Setup

To create a virtual environment and install the required dependencies, execute the following command:

```bash
bash install.sh
```

## Usage

### Build Knowledge Base with Image-Question-Answer samples

We build our knowledge base using some subsets of the dataset [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron), which contains 50 visual-language datasets (for training only). Run:

```bash
python kb/build_kb.py --datasets HuggingFaceM4/the_cauldron/<subset 1> HuggingFaceM4/the_cauldron/<subset 2> --save_dir <path to knowledge base>
```

- `--datasets`: A list of datasets to load into the knowledge base.
- `--save_dir`: The directory path where the samples, images, and embeddings of the knowledge base will be saved. If not specified, it defaults to `knowledge_base`.

### Server

TODO

### Client

TODO

## Evaluation

### Tasks

https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/v0.3.3/docs/current_tasks.md

### Run

```bash
export PYTHONPATH=./
python eval/ct3_lmms_eval.py \
  --model_path Qwen/Qwen2.5-VL-3B-Instruct \
  --task <task name> \
  --output_path <path to results> \
  --ttt \
  --kb_path <path to knowledge base> \
  --num_ttt_samples <number of ttt training samples> \
```


## Citation

If you find CT3+ code and paper helpful, please kindly cite:

```bibtex
@inproceedings{munoz2025CT3+,
  title = {CT3: Boosting Downstream Performance through Test-time Training on AI PCs with Remote Multi-Domain Knowledge Bases},
  ...
}
```
