
# A simple hook that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
            # self.hook = module.register_full_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

def initialize_module(m):
  if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.constant_(m.weight, 1)
    if m.bias is not None:
      nn.init.constant_(m.bias, 0)
  elif isinstance(m, nn.Linear):
    nn.init.normal_(m.weight, 0, 0.01)
    nn.init.constant_(m.bias, 0)