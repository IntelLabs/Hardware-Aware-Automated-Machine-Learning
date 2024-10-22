### Search on an existing super-network

If you have a trained super-network, you can start the search stage directly using the ```bootstrap_nas_search.py``` script located [here](https://github.com/openvinotoolkit/nncf/blob/develop/examples/experimental/torch/classification/bootstrap_nas_search.py).

You must pass the path where the weights and elasticity information have been stored, which is your log directory by default. 

```shell
python bootstrap_nas_search.py -m
train
--config <Config path to your config.json used when training the super-network> 
--log-dir <Path to your log dir for the search stage> 
--dataset
<cifar10, imagenet, or other, depending on your model>
--data <Path to your dataset>
--elasticity-state-path
 <Path to your last_elasticity.pth file generated when training of the super-network>
--supernet-weights <Path to your last_model_weights.pth generated during training of the super-network> 
--search-mode
```

#### Hardware-aware search 

BootstrapNAS can be made hardware-aware when searching for efficient sub-networks. To accomplish this, you can pass your own  `efficiency evaluator` for the target hardware to the search component.