import numpy as np

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