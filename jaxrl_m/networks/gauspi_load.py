import torch
import jax.numpy as jnp
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.networks.ps import GausPiNetwork, RobotModule, TaskModule
from jaxrl_m.vision import encoders
import jax
from jaxrl_m.utils.torch_to_flax import gauspi_flax_keys, torch_to_linen

checkpoint = torch.load(
    "/home/ksuresh/bridge_data_v2/Agent.pth",
    map_location=torch.device("cpu"),
)
gauspi_params = torch_to_linen(checkpoint["decoder"], gauspi_flax_keys, start_idx=0)

encoder_def = EncodingWrapper(
    encoder=encoders["mococonv5"](), use_proprio=True, stop_gradient=True
)
net = GausPiNetwork(
    encoder_def,
    TaskModule(
        hidden_dim=256,
        latent_interface_dim=128,
        anchors=jnp.load("/home/ksuresh/bridge_data_v2/anchor_states_sweep_embeddings.npy"),
    ),
    RobotModule(
        num_actions=7,
        hidden_dim=256,
        latent_interface_dim=128,
    ),
)

params = net.init(jax.random.PRNGKey(0), {"image": jnp.ones((1, 224, 224, 3)), "proprio": jnp.zeros((1, 7))})
breakpoint()