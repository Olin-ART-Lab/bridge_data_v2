from typing import Any, Callable, Sequence, Tuple
import torch
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np
from resnet_v1 import ResNetEncoder, BottleneckResNetBlock
from flax.core import FrozenDict
from pdb import set_trace as bp

import torchvision.models as models
from torch.nn.modules.linear import Identity

class MoCoConvEncoder(nn.Module):
    moco_checkpoint_path: str
    resnet: ResNetEncoder = ResNetEncoder

    def setup(self):
        pass

    def __call__(self, observations: jnp.ndarray):
        return self.resnet(observations)

# moco_conv_configs = {
#     "moco_conv3": {},
#     "moco_conv4": {},
#     "moco_conv5": {},
#     "moco_conv5_robocloud": MoCoConvEncoder(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock),
# }
# (stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)

def _get_flax_keys(keys, stage_sizes=[3, 4, 6, 3]):
    layerblock = None
    layer_idx = None
    if len(keys) == 2:  # first layer 
        layer, param = keys
        if layer == 'conv1':
            layer = 'conv_init'
        if layer == 'bn1':
            layer = 'norm_init'
    elif len(keys) == 3:
        layer, layer_idx, param = keys
        layer = f"Dense_{0 if layer_idx == '0' else 1}"
    elif len(keys) == 4:  # block layer
        layerblock, block_idx, layer, param = keys
        if "conv" in layer:
            layer = f"Conv_{int(layer[-1])-1}"
        if "bn" in layer:
            layer = f"BatchNorm_{int(layer[-1])-1}"
    elif len(keys) == 5:  # downsample layer
        layerblock, block_idx, layer, layer_idx, param = keys
        if layer_idx == '0':
            layer = 'conv_proj'
        if layer_idx == '1':
            layer = 'norm_proj'


    if param == 'weight':
        param = 'scale' if 'norm' in layer.lower() else 'kernel'
    if 'running' in param:
        param = 'mean' if 'mean' in param else 'var'


    if layerblock:
        prev_blocks = 0
        for i in range(int(layerblock[-1])-1):
            prev_blocks += stage_sizes[i]
        layerblock = f"BottleneckResNetBlock_{prev_blocks+int(block_idx)}"
        return [layerblock, layer, param]

    return [layer, param]

def add_to_params(params_dict, nested_keys, param, is_conv=False):
    if len(nested_keys) == 1:
        key, = nested_keys
        params_dict[key] = np.transpose(param, (2, 3, 1, 0)) if is_conv else np.transpose(param)
    else:
        assert len(nested_keys) > 1
        first_key = nested_keys[0]
        if first_key not in params_dict:
            params_dict[first_key] = {}
    
        add_to_params(params_dict[first_key], nested_keys[1:], param, ('Conv' in first_key and \
                                                                     nested_keys[-1] != 'bias'))

def torch_to_linen(state_dict, get_flax_keys):
    flax_params = {'params': {}, 'batch_stats': {}}
    for key, tensor in state_dict.items():
        keys = key.split('.')[2:]
        if len(keys):
            if keys[0] == 'fc':
                continue
            flax_keys = get_flax_keys(keys)
            if flax_keys[-1] is not None:
                if flax_keys[-1] in ('mean', 'var'):
                    add_to_params(flax_params['batch_stats'], flax_keys, tensor.detach().numpy())
                else:
                    add_to_params(flax_params['params'], flax_keys, tensor.detach().numpy())
    return flax_params

# Things to check:
# Types of the parameters
# Check the outputs of each block
# Maybe check the padding of the conv layers
# 
if __name__ == "__main__":
    checkpoint = torch.load('/home/ksuresh/bridge_data_checkpts/moco_conv5_robocloud.pth', map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    resnet_params = FrozenDict(torch_to_linen(state_dict, _get_flax_keys))
    batch = jnp.ones((1, 224, 224, 3))

    moco = ResNetEncoder(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock, norm="batch")

    out = moco.apply(resnet_params, batch)
    print(out.tolist()[0][:10])

    model = models.resnet50(pretrained=False, progress=False)
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    model.fc = Identity()

    # model.double()
    batch = np.ones((1, 3, 224, 224), dtype=np.float32)
    batch = torch.from_numpy(batch)
    print(model(batch).tolist()[0][:10])