from functools import partial
from typing import Any
import jax
import jax.numpy as jnp
from jaxrl_m.common.encoding import EncodingWrapper, GCEncodingWrapperPS
import numpy as np
import flax
import flax.linen as nn
import optax

from flax.core import FrozenDict
from jaxrl_m.common.typing import Batch
from jaxrl_m.common.typing import PRNGKey
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.networks.actor_critic_nets import Policy
from jaxrl_m.networks.mlp import MLP
from jaxrl_m.networks.ps import GausPiNetwork, TaskModule, RobotModule
from flax.core import freeze, unfreeze
from jaxrl_m.utils.torch_to_flax import torch_to_linen, resnet_flax_keys, gauspi_flax_keys

from flax.core import FrozenDict
import torch


class PSBCAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                params,
                (batch["observations"], batch["goals"]),
                train=True,
                name="actor",
                mutable=["batch_stats"],
                rngs={"dropout": key},
            )
            if type(dist) == tuple:
                dist = dist[0]
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            return (
                actor_loss,
                {
                    "actor_loss": actor_loss,
                    "mse": mse.mean(),
                    "log_probs": log_probs,
                    "pi_actions": pi_actions,
                    "mean_std": actor_std.mean(),
                    "max_std": actor_std.max(),
                },
            )

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn, pmap_axis=pmap_axis, has_aux=True
        )

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        dist, state = self.state.apply_fn(
            self.state.params,
            (observations, goals),
            train=False,
            name="actor",
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        # return dist, state
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist, state = self.state.apply_fn(
            self.state.params,
            (batch["observations"], batch["goals"]),
            train=False,
            name="actor",
            capture_intermediates=True,
            mutable=["intermediates"],
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        return {"mse": mse, "log_probs": log_probs, "pi_actions": pi_actions}

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        anchors: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        use_proprio: bool = False,
        network_kwargs: dict = {"hidden_dims": [256, 256]},
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "state_dependent_std": False,
            "dropout": 0.0,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
    ):
        encoder_def = GCEncodingWrapperPS(
            encoder=encoder_def, use_proprio=True, stop_gradient=True
        )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": GausPiNetwork(
                encoder_def,
                TaskModule(
                    hidden_dim=1028,
                    latent_interface_dim=128,
                    anchors=anchors,
                ),
                RobotModule(
                    num_actions=actions.shape[-1],
                    hidden_dim=256,
                    latent_interface_dim=128,
                ),
            )
        }

        model_def = ModuleDict(networks)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=0.0,
        )
        tx = optax.adam(lr_schedule)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(init_rng, actor=[(observations, goals)])
        checkpoint = torch.load(
            "/home/krishna/bridge_data_v2/moco_conv5_robocloud.pth",
            map_location=torch.device("cpu"),
        )
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            if not k.startswith('module.encoder_q'):
                del state_dict[k]
        resnet_params = torch_to_linen(state_dict, resnet_flax_keys)

        # checkpoint = torch.load(
        #     "/home/ksuresh/bridge_data_v2/Agent.pth",
        #     map_location=torch.device("cpu"),
        # )
        # gauspi_params = torch_to_linen(checkpoint["decoder"], gauspi_flax_keys, start_idx=0)
        params = unfreeze(params)

        params["params"]["modules_actor"]["encoder"]["encoder"] = resnet_params[
            "params"
        ]
        # params["params"]["modules_actor"]["task_module"] = gauspi_params[
        #     "params"
        # ]["task_module"]
        # params["params"]["modules_actor"]["robot_module"] = gauspi_params[
        #     "params"
        # ]["robot_module"]
        params["batch_stats"]["modules_actor"]["encoder"]["encoder"] = resnet_params[
            "batch_stats"
        ]
        # params["batch_stats"]["modules_actor"]["task_module"] = gauspi_params[
        #     "batch_stats"
        # ]["task_module"]
        # params["batch_stats"]["modules_actor"]["robot_module"] = gauspi_params[
        #     "batch_stats"
        # ]["robot_module"]

        params = freeze(params)

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        return cls(state, lr_schedule)
