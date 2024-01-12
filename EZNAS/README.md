# EZNAS: Evolving Zero-Cost Proxies For Neural Architecture Scoring

EZNAS is a genetic programming driven methodology for the automatic discovery of Zero-Cost Neural Architecture Scoring Metrics (ZC-NASMs). It aims to provide an interpretable, generalizable, and efficient approach to rank neural networks without the expensive training routines, significantly reducing the carbon footprint of Neural Architecture Search (NAS).

## Installation

Follow these steps to set up and run EZNAS:

### Step 1: Base Set-up
Run the provided setup_script.sh to install all necessary packages and dependencies.

```bash
bash setup_script.sh
```

This script should handle:

1. Installation of required Python packages.
2. Cloning of external GitHub repositories.
3. Setting up datasets and additional files necessary for running the project.

### Step 2: Set Environment Variable

Set the PROJ_HOME environment variable to the path of your project:

```bash
export PROJ_HOME="<Path to your project>"
```

### Step 3: Run evaluation

For SLURM based execution, modify runjob.sh as per server specification.

To reproduce results for a specific data-set, simply run the appropriate command in quotes from the reproduce.sh file.

```bash
python verify_scores.py --batch_size 16 --search_space NASBench201 --dataset cifar10 --nds_space ''
```

### Results

| Search Space           | Kendall τ    | Spearman ρ   |
|------------------------|--------------|--------------|
| NASBench-201 CIFAR-10  | 0.6195383854 | 0.8084988792 |
| NASBench-201 CIFAR-100 | 0.6168760649 | 0.7983379022 |
| NATSBench-SSS          | 0.7073727282 | 0.8873359833 |
| NDS DARTS              | 0.5466290384 | 0.7364709542 |
| NDS Amoeba             | 0.4130041903 | 0.5775007582 |
| NDS ENAS               | 0.5111310224 | 0.6932549307 |
| NDS PNAS               | 0.4781835008 |  0.656343803 |
| NDS NASNet             | 0.4312498051 | 0.6050820615 |


Note that the above table is for a batch size of 16. For better results, a higher batch-size is recommended! For instance, for NATSBench-SS at batch-size of 64, the Spearman ρ is 0.91.

# Citation

If you use the code or data in your research, please use the following BibTex entry:

```
@inproceedings{
akhauri2022eznas,
title={{EZNAS}: Evolving Zero-Cost Proxies For Neural Architecture Scoring},
author={Yash Akhauri and Juan Pablo Munoz and Nilesh Jain and Ravishankar Iyer},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=lSqaDG4dvdt}
}
```