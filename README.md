# Hardware-Aware Automated Machine Learning Research

This repository contains instructions and examples for efficient neural architecture discovery and optimization solutions developed at Intel Labs. 

### :fire:[SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](./SQFT/README.md)

SQFT is a solution for fine-tuning low-precision and sparse large models using parameter-efficient fine-tuning (PEFT). It includes an innovative strategy that enables the merging of sparse weights with low-rank adapters without losing sparsity and accuracy, overcoming the limitations of previous approaches. SQFT also addresses the challenge of having quantized weights and adapters with different numerical precisions, enabling merging in the desired numerical format without sacrificing accuracy.

SQFT fine-tunes **sparse** and **low-precision** LLMs using parameter-efficient techniques, merging sparse weights with low-rank adapters while maintaining sparsity and accuracy, and handling quantized weights and adapters of different precisions.

Shears integrates cost-effective
sparsity and Neural Low-rank adapter
Search (NLS) to  further improve the efficiency of Parameter-Efficient Fine-Tuning (PEFT) approaches. 



### [NNCF's BootstrapNAS - Notebooks and Examples](./BootstrapNAS/README.md) 

BootstrapNAS automates the generation of weight-sharing super-networks using the Neural Network Compression Framework (NNCF). 

### [LoNAS: Elastic Low-Rank Adapters for Efficient Large Language Models](./LoNAS/README.md) 

This is an initial exploration of using weight-sharing NAS for the compression of large language models. We explore a search space of elastic low-rank adapters while reducing full-scale NAS's memory and compute requirements. This results in high-performing compressed models obtained from weight-sharing super-networks. We investigate the benefits and limitations of this method, motivating follow-up work.

### [EFTNAS: Searching for Efficient Language Models in First-Order Weight-Reordered Super-Networks](./EFTNAS/README.md)

Integrating neural architecture search (NAS) and network pruning techniques, we effectively generate and train
weight-sharing super-networks that contain efficient, high-performing, and compressed transformer-based models.
A common challenge in NAS is designing the search space, for which we propose a method to automatically
obtain the boundaries of the search space and then derive the rest of the intermediate possible architectures using
a first-order weight importance technique. The proposed end-to-end NAS solution, EFTNAS, discovers efficient
subnetworks that have been compressed and fine-tuned for downstream NLP tasks.

### [EZNAS: Evolving Zero-Cost Proxies For Neural Architecture Scoring](./EZNAS/README.md) 

EZNAS is a genetic programming-driven methodology for automatically discovering Zero-Cost Neural Architecture Scoring Metrics (ZC-NASMs).

