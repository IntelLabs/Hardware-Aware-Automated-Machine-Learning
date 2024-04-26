#!/bin/bash
set -e
set -x

SHEARS_PATH=$PWD

mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.31.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/transformers-modifications-for-shears-usage.patch && pip install -e . && cd ..

# peft
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.5.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-modifications-for-shears-usage.patch && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-modifications-for-shears-inference-usage.patch && pip install -e . && cd ..

# nncf
git clone https://github.com/openvinotoolkit/nncf.git
cd nncf && git checkout 544d5141 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/nncf-modifications-for-shears-usage.patch && pip install -e . && cd ..

# others
pip install datasets accelerate sentencepiece protobuf
