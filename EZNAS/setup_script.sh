python -m pip install pyyaml==5.4.1 absl-py matplotlib deap tensorflow==1.15 networkx toolz xautodl torchsummary simplejson nats_bench

# External libraries
git clone git@github.com:google-research/nasbench.git
cd nasbench/
python -m pip install -e .
cd ./..
git clone git@github.com:D-X-Y/NAS-Bench-201.git

# Residual imports required
mkdir residual_imports
cd residual_imports
git clone git@github.com:BayesWatch/nas-without-training.git
cp -r nas-without-training/pycls ./..
cp -r nas-without-training/models ./..
cp -r nas-without-training/config_utils ./..
cp -r nas-without-training/nas_101_api ./..
cd ./..
rm -rf residual_imports

# NDS Dataset
wget https://dl.fbaipublicfiles.com/nds/data.zip
unzip data.zip
rm data.zip

# NASBench-201 and NATSBench-SSS zip files
python -m pip uninstall gdown
python -m pip install -U --no-cache-dir gdown --pre
mkdir dataset_generator
cd dataset_generator
gdown 16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_
gdown 1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX
cd ./..