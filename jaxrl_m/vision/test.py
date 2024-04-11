import jax
import jax.numpy as jnp
import numpy as np
import torch

from flax import linen as nn


def to_numpy(x):
    return x.detach().cpu().numpy()


checkpoint = torch.load(
    "/home/ksuresh/bridge_data_checkpts/moco_conv5_robocloud.pth",
    map_location=torch.device("cpu"),
)
state_dict = checkpoint["state_dict"]
pyt_conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
pyt_conv.weight = torch.nn.Parameter(state_dict["module.encoder_q.conv1.weight"])
pyt_norm = torch.nn.BatchNorm2d(64)
pyt_norm.bias = torch.nn.Parameter(state_dict["module.encoder_q.bn1.bias"])
pyt_norm.weight = torch.nn.Parameter(state_dict["module.encoder_q.bn1.weight"])
pyt_norm.running_mean = torch.nn.Parameter(
    state_dict["module.encoder_q.bn1.running_mean"], requires_grad=False
)
pyt_norm.running_var = torch.nn.Parameter(
    state_dict["module.encoder_q.bn1.running_var"], requires_grad=False
)
pyt_norm.eval()

jax_conv = nn.Conv(
    features=64,
    kernel_size=(7, 7),
    strides=(2, 2),
    padding=[(3, 3), (3, 3)],
    use_bias=False,
    name="jax_conv",
)
jax_norm = nn.BatchNorm(momentum=0.9, use_running_average=True)

# Load weights from PyTorch to JAX
variables = {"params": {"kernel": jnp.transpose(to_numpy(pyt_conv.weight), (2, 3, 1, 0))}}
variables_norm = {
    "params": {
        "scale": to_numpy(pyt_norm.weight),
        "bias": to_numpy(pyt_norm.bias),
    },
    "batch_stats": {
        "mean": to_numpy(pyt_norm.running_mean),
        "var": to_numpy(pyt_norm.running_var),
    },
}

# Input to the models
x = np.asarray(jnp.ones((1, 224, 224, 3)), dtype=jnp.float32)

# Forward pass PyTorch
py_out = pyt_conv(torch.from_numpy(x).permute(0, 3, 1, 2))
py_out = pyt_norm(py_out)
py_out = to_numpy(py_out)

# Forward pass JAX
jax_out = jax_conv.apply(variables, x)
jax_out = jax_norm.apply(variables_norm, jax_out)

# Transpose the output to match the PyTorch output
jax_out = np.transpose(jax_out, (0, 3, 1, 2))
np.testing.assert_almost_equal(py_out, jax_out, decimal=4)
