#!/bin/bash
set -e
set -x

LONAS_PATH=$PWD
mkdir third_party && cd third_party

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers && git checkout v4.31.0 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/transformers-v4.31.0.patch && pip install -e . && cd ..

# peft
git clone https://github.com/huggingface/peft.git
cd peft && git checkout v0.5.0 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/peft-v0.5.0.patch && pip install -e . && cd ..

# nncf
git clone https://github.com/openvinotoolkit/nncf.git nncf
cd nncf && git checkout 544d5141 && git apply --ignore-space-change --ignore-whitespace $LONAS_PATH/patches/nncf-544d5141.patch && pip install -e . && cd ..

# others
pip install datasets accelerate sentencepiece protobuf evaluate
