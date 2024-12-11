#!/bin/bash
set -e
set -x

MULTIPRUNER_PATH=$PWD
mkdir third_party && cd third_party

pip install 'numpy<2.0.0' setuptools==69.5.1

mkdir third_party && cd third_party
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.42.4 && git apply --ignore-space-change --ignore-whitespace ${MULTIPRUNER_PATH}/patches/transformers-v4.42.4.patch && pip install -e . && cd ..

pip install datasets accelerate sentencepiece protobuf bitsandbytes
pip install lm-eval==0.4.2
