# SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs

Official implementation of SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs.

This repo contains the code for **SparAMX**, a set of open-source customized sparse kernels that can speed up any PyTorch model by automatically replacing all linear layers with our customized layer. Furthermore, we demonstrate for the first time the use of unstructured sparsity in the attention computation and achieving \textbf{1.14}$\times$ speedup over the current systems without compromising accuracy.

| Stock PyTorch | SparAMX |
|:-----------:|:-----------:|
| <img src="Videos/stock.gif" alt="Stock" width="100%"> | <img src="Videos/sparamx.gif" alt="TLD" width="100%"> |

# torch-custom-linear
Custom implementation of linear through torch extension

### Dependency
```pip install -r requirements.txt```

### Build & Install Custom Kernel
```
python setup.py install
```

### Run Experiments Example

Please make sure you're logged in to HuggingFace through the CLI if you'll be using a private model.

You need to define the experiments you want to run in `generate_experiments.py` then run 
```
python generate_experiments.py
```

A file `experiments.csv` is generated. Modify it if needed. After that run
```
./run_experiments.sh
```

Your results will be saved inside folder `experiment_results/YYYY-MM-DD_HH-MM-SS`.

## Citation
If you find our SparAMX code and paper helpful, please kindly cite:
```bibtex
@misc{abouelhamayed2025sparamxacceleratingcompressedllms,
      title={SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs}, 
      author={Ahmed F. AbouElhamayed and Jordan Dotzel and Yash Akhauri and Chi-Chih Chang and Sameh Gobriel and J. Pablo Muñoz and Vui Seng Chua and Nilesh Jain and Mohamed S. Abdelfattah},
      year={2025},
      eprint={2502.12444},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.12444}, 
}
'''

