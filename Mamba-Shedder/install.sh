#!/bin/bash
set -e
set -x

MAMBA_SHEDDER_PATH=$PWD

pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch==2.4.0
pip install transformers==4.43.0
pip install causal-conv1d>=1.4.0

mkdir -pv ${MAMBA_SHEDDER_PATH}/third_party

pushd ${MAMBA_SHEDDER_PATH}/third_party
git clone https://github.com/state-spaces/mamba.git
pushd mamba
git checkout 62db608
git apply --ignore-space-change --ignore-whitespace ${MAMBA_SHEDDER_PATH}/patches/mamba-62db608.patch
pip install .
pushd ${MAMBA_SHEDDER_PATH}

pip install lm-eval==0.4.2
echo "Environment all ready.  execute 'source venv/bin/activate' to run"
