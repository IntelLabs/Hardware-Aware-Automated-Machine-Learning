#!/bin/bash
set -x
set -e

SPARSITY=$1

BASE_MODEL_PATH=mistralai/Mistral-7B-v0.3
SPARSE_BASE_MODEL_PATH=sqft-mistral-7b-v0.3-${SPARSITY}-base
QUANT_BASE_MODEL_PATH=sqft-mistral-7b-v0.3-${SPARSITY}-base-gptq

python wanda/main.py --model ${BASE_MODEL_PATH} --prune_method wanda --sparsity_ratio $(echo "scale=2; ${SPARSITY}/100" | bc) --sparsity_type unstructured --save wanda_out --save_model ${SPARSE_BASE_MODEL_PATH}
python utils/quantization.py --base_model_path ${SPARSE_BASE_MODEL_PATH} --output_dir ${QUANT_BASE_MODEL_PATH}
