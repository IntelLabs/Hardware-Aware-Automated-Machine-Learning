#!/bin/bash
set -x
set -e

SPARSITY=$1

BASE_MODEL_PATH=meta-llama/Meta-Llama-3-8B
SPARSE_BASE_MODEL_PATH=sqft-llama-3-8b-${SPARSITY}-base
QUANT_BASE_MODEL_PATH=sqft-llama-3-8b-${SPARSITY}-base-gptq

python wanda/main.py --model ${BASE_MODEL_PATH} --prune_method wanda --sparsity_ratio $(echo "scale=2; ${SPARSITY}/100" | bc) --sparsity_type unstructured --save wanda_out --save_model ${SPARSE_BASE_MODEL_PATH}
python utils/quantization.py --base_model_path ${SPARSE_BASE_MODEL_PATH} --output_dir ${QUANT_BASE_MODEL_PATH}
