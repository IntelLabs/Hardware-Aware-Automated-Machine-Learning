#!/bin/bash
set -e
set -x

MULTIPRUNER_PATH=$PWD

python3.10 -m venv venv
source venv/bin/activate

mkdir -pv third_party
pushd third_party

git clone https://github.com/huggingface/transformers.git
pushd transformers
git checkout v4.45.0
git apply --ignore-space-change --ignore-whitespace ${MULTIPRUNER_PATH}/patches/transformers-v4.45.0.patch
pip install -e .

pushd ${MULTIPRUNER_PATH}

pip install -r requirements.txt

echo "Environment all ready.  execute 'source venv/bin/activate' to run"

