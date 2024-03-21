# BootstrapNAS Jupyter Notebooks

---

<p align="center">
<img src="architecture.png" alt="BootstrapNAS Architecture" width="500"/>
</p>

BootstrapNAS (1) takes as input a pre-trained model. (2) It uses this model to generate a weight-sharing super-network. (3) BootstrapNAS then applies a training strategy, and once the super-network has been trained, (4) it searches for efficient subnetworks that satisfy the user's requirements. (5) The configuration of the discovered sub-network(s) is returned to the user.

## Quickstart 

Please follow the instructions [here](https://github.com/jpablomch/bootstrapnas/wiki/Quickstart).

If you already have a super-network trained with BootstrapNAS, please follow the instructions to search for sub-networks [here](https://github.com/jpablomch/bootstrapnas/wiki/Subnetwork_Search).

More information about BootstrapNAS is available in our papers:

[Automated Super-Network Generation for Scalable Neural Architecture Search](https://openreview.net/attachment?id=HK-zmbTB8gq&name=main_paper_and_supplementary_material).

```bibtex
  @inproceedings{
    munoz2022automated,
    title={Automated Super-Network Generation for Scalable Neural Architecture Search},
    author={Mu{\~{n}}oz, J. Pablo and Lyalyushkin, Nikolay and Lacewell, Chaunte and Senina, Anastasia and Cummings, Daniel and Sarah, Anthony  and Kozlov, Alexander and Jain, Nilesh},
    booktitle={First Conference on Automated Machine Learning (Main Track)},
    year={2022},
    url={https://openreview.net/forum?id=HK-zmbTB8gq}
  }
```
[Enabling NAS with Automated Super-Network Generation](https://arxiv.org/abs/2112.10878)

```BibTex
@article{
  bootstrapNAS,
  author    = {Mu{\~{n}}oz, J. Pablo  and Lyalyushkin, Nikolay  and Akhauri, Yash and Senina, Anastasia and Kozlov, Alexander  and Jain, Nilesh},
  title     = {Enabling NAS with Automated Super-Network Generation},
  journal   = {CoRR},
  volume    = {abs/2112.10878},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10878},
  eprinttype = {arXiv},
  eprint    = {2112.10878},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10878.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contributing to BootstrapNAS
Please follow the contribution guidelines in [NNCF](https://github.com/openvinotoolkit/nncf). 

