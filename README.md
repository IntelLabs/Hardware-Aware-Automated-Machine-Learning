# Hardware-Aware Automated Machine Learning Research

This repository contains instructions and examples for efficient neural architecture discovery and optimization solutions developed at Intel Labs. 

### [EZNAS: Evolving Zero-Cost Proxies For Neural Architecture Scoring](./EZNAS/README.md) 

EZNAS is a genetic programming-driven methodology for automatically discovering Zero-Cost Neural Architecture Scoring Metrics (ZC-NASMs).

### [NNCF's BootstrapNAS - Notebooks and Examples](./BootstrapNAS/README.md) 

BootstrapNAS automates the generation of weight-sharing super-networks using the Neural Network Compression Framework (NNCF). 

### [LoNAS: Elastic Low-Rank Adapters for Efficient Large Language Models](./LoNAS/README.md) 

Initial exploration of using NAS on large language models by exploring a search space of elastic low-rank adapters
while reducing memory and compute requirements of full-scale NAS, resulting in high-performing compressed models obtained from
weight-sharing super-networks. We investigate the benefits and limitations of this method, motivating follow-up work.

### [EFTNAS: Searching for Efficient Language Models in First-Order Weight-Reordered Super-Networks](./EFTNAS/README.md)

Integrating neural architecture search (NAS) and network pruning techniques, we effectively generate and train
weight-sharing super-networks that contain efficient, high-performing, and compressed transformer-based models.
A common challenge in NAS is the design of the search space, for which we propose a method to automatically
obtain the boundaries of the search space and then derive the rest of the intermediate possible architectures using
a first-order weight importance technique. The proposed end-to-end NAS solution, EFTNAS, discovers efficient
subnetworks that have been compressed and fine-tuned for downstream NLP tasks.
