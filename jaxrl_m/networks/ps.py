from typing import Callable, Optional, Sequence
import distrax
import flax.linen as nn
import jax.numpy as jnp
from jaxrl_m.common.common import default_init

LOG_SIG_MAX = 2
LOG_SIG_MIN = -9
epsilon = 1e-7  # 1e-6

class TaskModule(nn.Module):
    hidden_dim: int
    latent_interface_dim: int # this is the dim of the space between the task and robot modules
    anchors: jnp.ndarray
    use_anchors: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=1)

        if self.use_anchors:
            task_anchor_state = jnp.concatenate([x, self.anchors.transpose()], axis=1).transpose()
            x = nn.BatchNorm(momentum=0.9, use_running_average=not train, name="norm_input")(task_anchor_state)
        else:
            x = nn.BatchNorm(momentum=0.9, use_running_average=not train, name="norm_input")(x)
        x = nn.Dense(self.hidden_dim, name="linear1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="linear2")(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_interface_dim, name="linear3")(x)
        x = nn.normalize(x)
        if not self.use_anchors:
            return x
        x_task = x[0:-self.latent_interface_dim, :]
        x_anchor = x[-self.latent_interface_dim:, :]
        relative_interface = jnp.matmul(x_task, x_anchor.T)
        return relative_interface

class RobotModule(nn.Module):
    num_actions: int
    latent_interface_dim: int # this is the dim of the space between the task and robot modules
    hidden_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray, robot_state: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        action_emb = nn.BatchNorm(momentum=0.9, use_running_average=not train, name="norm_input")(robot_state)
        action_emb = nn.Dense(self.hidden_dim - self.latent_interface_dim, kernel_init=default_init(), name="linearR")(action_emb)
        if len(action_emb.shape) == 1:
            action_emb = jnp.expand_dims(action_emb, axis=0)
        x = jnp.concatenate([action_emb, x], axis=1)
        x = nn.Dense(self.hidden_dim, name="linear1")(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name="linear2")(x)
        x = nn.relu(x)
        # x = nn.Dense(35, name="linear3")(x)
        # return x
        mean = nn.Dense(self.num_actions, name="action_mean")(x)
        log_std = nn.Dense(self.num_actions, name="action_log_std")(x)

        log_std = jnp.clip(log_std, a_min=LOG_SIG_MIN, a_max=LOG_SIG_MAX)
        
        distribution = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std)
        )

        return distribution


class GausPiNetwork(nn.Module):
    encoder: nn.Module
    task_module: TaskModule
    robot_module: RobotModule

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        robot_state = x["proprio"]
        x = self.encoder(x, train=train)
        relative_interface = self.task_module(x, train=train)
        dist = self.robot_module(relative_interface, robot_state, train=train)
        return dist