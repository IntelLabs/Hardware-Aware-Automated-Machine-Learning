#!/bin/bash
set -e
set -x

SHEARS_PATH=$PWD

# pytorch
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
pushd transformers && git checkout v4.31.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/transformers-modifications-for-shears-usage.patch && pip install -e . && popd

# peft
git clone https://github.com/huggingface/peft.git peft
pushd peft && git checkout v0.5.0 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-modifications-for-shears-usage.patch && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/peft-modifications-for-shears-inference-usage.patch && pip install -e . --no-deps && popd

# nncf
git clone https://github.com/openvinotoolkit/nncf.git nncf
pushd nncf && git checkout 544d5141 && git apply --ignore-space-change --ignore-whitespace $SHEARS_PATH/patches/nncf-modifications-for-shears-usage.patch && pip install -e . && popd

# others
pip install datasets==2.14.5 accelerate sentencepiece protobuf
