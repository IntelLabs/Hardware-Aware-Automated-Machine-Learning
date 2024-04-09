#!/bin/bash
set -e
set -x

EFTNAS_PATH=$PWD

git clone https://github.com/openvinotoolkit/nncf.git nncf_eftnas
cd nncf_eftnas
git checkout 415c3c4d
# Apply Patch
git apply $EFTNAS_PATH/patches/nncf.patch
pip install -e .
cd ..


git clone https://github.com/huggingface/transformers.git transformers_eftnas
cd transformers_eftnas
git checkout v4.29.1
git apply $EFTNAS_PATH/patches/transformers.patch
pip install -e .
pip install -r examples/pytorch/text-classification/requirements.txt
cd ..
