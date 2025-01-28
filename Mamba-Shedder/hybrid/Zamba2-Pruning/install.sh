#!/bin/bash
set -e
set -x

MAMBA_SHEDDER_ZAMBA2_PATH=$PWD

cp ../../utils.py .
cp ../../patches/mamba-62db608.patch ./patches

pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==2.4.0
pip install causal-conv1d>=1.4.0

mkdir -pv ${MAMBA_SHEDDER_PATH}/third_party

pushd ${MAMBA_SHEDDER_ZAMBA2_PATH}/third_party
git clone https://github.com/state-spaces/mamba.git
pushd mamba
git checkout 62db608
git apply --ignore-space-change --ignore-whitespace ${MAMBA_SHEDDER_ZAMBA2_PATH}/patches/mamba-62db608.patch
pip install .
pip install lm-eval==0.4.2

pushd ${MAMBA_SHEDDER_ZAMBA2_PATH}/third_party
git clone https://github.com/Zyphra/transformers_zamba2.git
cd transformers_zamba2
git checkout 7593823
git apply --ignore-space-change --ignore-whitespace ${MAMBA_SHEDDER_ZAMBA2_PATH}/patches/zamba2-7593823.patch
pip install -e . --no-deps
pip install tokenizers==0.19.0 numpy==1.26.4 accelerate
pushd ${MAMBA_SHEDDER_ZAMBA2_PATH}

echo "Environment all ready.  execute 'source venv/bin/activate' to run"
