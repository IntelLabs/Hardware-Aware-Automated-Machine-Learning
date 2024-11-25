#!/bin/bash
set -e
set -x

pip install 'numpy<2.0.0' setuptools==69.5.1 wheel

SQFT_PATH=$PWD
mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.44.2 && git apply --ignore-space-change --ignore-whitespace ${SQFT_PATH}/patches/transformers-v4.44.2.patch && pip install -e . && cd ..

# peft
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.10.0 && git apply --ignore-space-change --ignore-whitespace ${SQFT_PATH}/patches/peft-v0.10.0.patch && pip install -e . && cd ..

pip install datasets accelerate sentencepiece protobuf
pip install optimum==1.18.0 --no-deps
pip install git+https://github.com/AutoGPTQ/AutoGPTQ@866b4c8

# nncf
git clone https://github.com/openvinotoolkit/nncf.git
cd nncf && git checkout f143e1c && git apply --ignore-space-change --ignore-whitespace ${SQFT_PATH}/patches/nncf-f143e1c.patch && pip install -e . && cd ..

# lm-eval-harness
pip install lm-eval==0.4.2
