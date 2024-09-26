#!/bin/bash
set -x
set -e

SPARSITY=$1

BASE_MODEL_PATH=microsoft/Phi-3-mini-4k-instruct
SPARSE_BASE_MODEL_PATH=sqft-phi-3-mini-4k-${SPARSITY}-base
QUANT_BASE_MODEL_PATH=sqft-phi-3-mini-4k-${SPARSITY}-base-gptq

python wanda/main.py --model ${BASE_MODEL_PATH} --prune_method wanda --sparsity_ratio $(echo "scale=2; ${SPARSITY}/100" | bc) --sparsity_type unstructured --save wanda_out --save_model ${SPARSE_BASE_MODEL_PATH}
python utils/quantization.py --base_model_path ${SPARSE_BASE_MODEL_PATH} --output_dir ${QUANT_BASE_MODEL_PATH}
