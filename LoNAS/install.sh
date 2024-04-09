#!/bin/bash
set -e
set -x

LONAS_PATH=$PWD

mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
pushd transformers && git checkout v4.31.0 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/transformers-modifications-for-lonas-usage.patch && pip install -e . && popd

# peft
git clone https://github.com/huggingface/peft.git
pushd peft && git checkout v0.5.0 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/peft-modifications-for-lonas-usage.patch && pip install -e . && popd

# nncf
git clone https://github.com/openvinotoolkit/nncf.git nncf
pushd nncf && git checkout 544d5141 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/nncf-modifications-for-lonas-usage.patch && pip install -e . &&

# others
pip install datasets accelerate sentencepiece protobuf
