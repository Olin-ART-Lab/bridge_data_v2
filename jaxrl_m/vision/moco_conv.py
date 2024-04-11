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
        add_to_params(params_dict[first_key], nested_keys[1:], param, ('conv' in first_key.lower() and \
                                                                     nested_keys[-1] != 'bias'))

def torch_to_linen(state_dict, get_flax_keys):
    flax_params = {'params': {}, 'batch_stats': {}}
    for key, tensor in state_dict.items():
        if not key.startswith('module.encoder_q'):
            continue
        keys = key.split('.')[2:]
        if len(keys):
            if keys[0] == 'fc':
                continue
            flax_keys = get_flax_keys(keys)
            if flax_keys[-1] is not None:
                if flax_keys[-1] in ('mean', 'var'):
                    add_to_params(flax_params['batch_stats'], flax_keys, tensor.detach().cpu().numpy())
                else:
                    add_to_params(flax_params['params'], flax_keys, tensor.detach().cpu().numpy())
    return flax_params

if __name__ == "__main__":
    checkpoint = torch.load('/home/ksuresh/bridge_data_checkpts/moco_conv5_robocloud.pth', map_location=torch.device('cpu'))
    resnet_params = FrozenDict(torch_to_linen(checkpoint['state_dict'], _get_flax_keys))

    state_dict = checkpoint['state_dict']
    batch = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)

    moco = ResNetEncoder(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock, norm="batch", moco=True)
    out, state = moco.apply(resnet_params, batch, train = False, capture_intermediates=True, mutable=["intermediates"])
    intermediates = state['intermediates']
    

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

    ## Checking if first conv layer is same    
    jax_conv1 = resnet_params['params']['conv_init']['kernel']
    pyt_conv1 = state_dict['conv1.weight']
    pyt_conv1 = np.transpose(pyt_conv1, (2, 3, 1, 0))
    np.testing.assert_almost_equal(pyt_conv1, jax_conv1, decimal=6)

    jax_bn1 = resnet_params['params']['norm_init']['bias']
    pyt_bn1 = state_dict['bn1.bias']
    pyt_bn1 = np.transpose(pyt_bn1)
    np.testing.assert_almost_equal(pyt_bn1, jax_bn1, decimal=6)

    jax_bn1 = resnet_params['batch_stats']['norm_init']['var']
    pyt_bn1 = state_dict['bn1.running_var']
    pyt_bn1 = np.transpose(pyt_bn1)
    np.testing.assert_almost_equal(pyt_bn1, jax_bn1, decimal=6)

    msg = model.load_state_dict(state_dict, strict=False)
    model.fc = Identity()
    model.eval()
    batch = np.ones((1, 3, 224, 224), dtype=np.float32)
    batch = torch.from_numpy(batch)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # model.conv1.register_forward_hook(get_activation('conv1'))
    # output = model(batch)

    # jax_out = intermediates['conv_init']['__call__'][0]
    # pyt_out = activation['conv1'].numpy()
    # jax_out = np.transpose(jax_out, (0, 3, 1, 2))
    # np.testing.assert_almost_equal(pyt_out, jax_out, decimal=4)
    # model.maxpool.register_forward_hook(get_activation('out'))
    # model.layer4[2].bn3.register_forward_hook(get_activation('out'))
    # output = model(batch)
    # jax_out = intermediates['BottleneckResNetBlock_0']['BatchNorm_0']['__call__'][0]
    # jax_out = intermediates['BottleneckResNetBlock_15']['c'][0]
    # pyt_out = activation['out'].numpy()
    # jax_out = np.transpose(jax_out, (0, 3, 1, 2))
    # np.testing.assert_almost_equal(pyt_out, jax_out, decimal=4)
    
    ## Check embedding layer
    np.testing.assert_almost_equal(out, model(batch).detach().numpy(), decimal=5)