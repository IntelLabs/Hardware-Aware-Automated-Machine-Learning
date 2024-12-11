#!/bin/bash
set -e
set -x

pip install 'numpy<2.0.0' setuptools==69.5.1 wheel
pip install transformers==4.47.0

# peft
SQFT_PATH=$PWD
mkdir third_party && cd third_party
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.10.0 && git apply --ignore-space-change --ignore-whitespace ${SQFT_PATH}/patches/peft-v0.10.0.patch && pip install -e . && cd ..

pip install datasets accelerate sentencepiece protobuf
pip install optimum --no-deps
pip install git+https://github.com/AutoGPTQ/AutoGPTQ@866b4c8

# lm-eval-harness
pip install lm-eval==0.4.2
