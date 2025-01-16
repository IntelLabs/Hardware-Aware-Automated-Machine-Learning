# Model Optimization Research 🚀

Welcome to the repository that showcases advanced neural architecture discovery and optimization solutions from Intel Labs. Here, you'll find cutting-edge research papers and their corresponding code implementations, all aimed at pushing the boundaries of model efficiency and performance.

## Featured Research Papers 📚

### MultiPruner: Fine-Grained Training-Free Structure Removal in Foundation Models
**Authors:** J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain  
**Links:** [Paper]() | [Code](./MultiPruner)  

MultiPruner is a training-free pruning approach for large pre-trained models that iteratively compresses residual blocks, MLP channels, and MHA heads, achieving superior zero-shot accuracy and model compression.

---

### SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models
**Authors:** J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain  
**Conference:** EMNLP 2024 Findings  
**Links:** [Paper](https://arxiv.org/abs/2410.03750) | [Code](./SQFT)  

SQFT fine-tunes **sparse** and **low-precision** LLMs using parameter-efficient techniques, merging sparse weights with low-rank adapters while maintaining sparsity and accuracy, and handling quantized weights and adapters of different precisions.

---

### SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs

SparAMX utilizes AMX support on the latest Intel CPUs along with unstructured sparsity to achieve a reduction in _end-to-end_ latency compared to the current PyTorch implementation by applying our technique in linear layers. 

**Authors:** Ahmed F. AbouElhamayed, Jordan Dotzel, Yash Akhauri, Chi-Chih Chang, Sameh Gobriel, J. Pablo Munoz, Vui Seng Chua, Nilesh Jain, Mohamed S. Abdelfattah  
**Links:** [Paper]() | [Code](./SparAMX)

---

### Shears: Unstructured Sparsity with Neural Low-rank Adapter Search
**Authors:** J. Pablo Muñoz, Jinjie Yuan, Nilesh Jain  
**Conference:** NAACL 2024 (Industry Track)  
**Links:** [Paper](https://arxiv.org/abs/2404.10934) | [Code](./Shears)  

Shears integrates cost-effective **sparsity** and **Neural Low-rank adapter Search (NLS)** to further improve the efficiency of Parameter-Efficient Fine-Tuning (PEFT) approaches.

---

### LoNAS: Elastic Low-Rank Adapters for Efficient Large Language Models
**Authors:** J. Pablo Muñoz, Jinjie Yuan, Yi Zheng, Nilesh Jain  
**Conference:** LREC-COLING 2024  
**Links:** [Paper](https://aclanthology.org/2024.lrec-main.940) | [Code](./LoNAS)  

LoNAS explores weight-sharing NAS for compressing large language models using elastic low-rank adapters, achieving high-performing models balancing efficiency and performance.

---

### EFTNAS: Searching for Efficient Language Models in First-Order Weight-Reordered Super-Networks
**Authors:** J. Pablo Muñoz, Yi Zheng, Nilesh Jain  
**Conference:** LREC-COLING 2024  
**Links:** [Paper](https://aclanthology.org/2024.lrec-main.497) | [Code](./EFTNAS)  

EFTNAS integrates neural architecture search and network pruning to automatically generate and train efficient, high-performing, and compressed transformer-based models for NLP tasks.

---

### EZNAS: Evolving Zero Cost Proxies For Neural Architecture Scoring
**Authors:** Yash Akhauri, J. Pablo Muñoz, Nilesh Jain, Ravi Iyer  
**Conference:** NeurIPS 2022   
**Links:** [Paper](https://arxiv.org/abs/2209.07413) | [Code](./EZNAS)  

EZNAS is a genetic programming-driven methodology for automatically discovering Zero-Cost Neural Architecture Scoring Metrics (ZC-NASMs).

---

### BootstrapNAS: Enabling NAS with Automated Super-Network Generation  
**Authors:** J. Pablo Muñoz, Nikolay Lyalyushkin, Yash Akhauri, Anastasia Senina, Alexander Kozlov, Chaunte Lacewell, Daniel Cummings, Anthony Sarah, Nilesh Jain  
**Conferences:** AutoML 2022 (Main Track), AAAI 2022 (Practical Deep Learning in the Wild)  
**Links:** [Paper AutoML](https://proceedings.mlr.press/v188/munoz22a) [Paper AAAI](https://arxiv.org/pdf/2112.10878)| [Code](./BootstrapNAS)  

BootstrapNAS generates weight-sharing super-networks from pre-trained models, and discovers efficient subnetworks.


## Additional Resources 📂

### NNCF's BootstrapNAS - Notebooks and Examples

Explore practical examples and notebooks related to NNCF's BootstrapNAS, a tool designed to facilitate neural architecture search and optimization.  
**Links:** [Code](./BootstrapNAS)

---

We hope you find these resources valuable for your research and development in the field of model optimization. Happy exploring! 🌟
