# ResNet50-CIFAR10

## 1. Current Results

<center>

| Architecture | MACs | Acc@1 | 
|-|-|-|
[Pretrained](train_results_others/pretrained.pt)        |  325.8M  |  93.65  |
[SuperNet](supernet_weights.pth)          |  325.8M  |  94.09  |
Minimum SubNet    |  64.1M   |  92.61  |
[Best Found SubNet](search_results_others/subnetwork_best.pth) |  68.3M   |  92.96  | 

</center>


## 2. Search Progression
<p align="center">
<img src="search_progression.png" alt="resnet50-cifar10 search progression" width="500"/>
</p>