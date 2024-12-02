#!/bin/bash
set -e
set -x

SHEARS_PATH=$PWD
mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.31.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/transformers-v4.31.0.patch && pip install -e . && cd ..

# peft
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.5.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-v0.5.0.patch && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-v0.5.0-inference.patch && pip install -e . && cd ..

# nncf
git clone https://github.com/openvinotoolkit/nncf.git
cd nncf && git checkout 544d5141 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/nncf-544d5141.patch && pip install -e . && cd ..

# others
pip install datasets accelerate sentencepiece protobuf
