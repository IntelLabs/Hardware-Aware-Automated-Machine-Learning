#!/bin/bash
set -e
set -x

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

echo "Environment all ready.  execute 'source venv/bin/activate' to run"
