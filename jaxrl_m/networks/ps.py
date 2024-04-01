from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
from jaxrl_m.common.common import default_init

LOG_SIG_MAX = 2
LOG_SIG_MIN = -9
epsilon = 1e-7  # 1e-6

class TaskModule(nn.Module):
    task_input_dim: int
    hidden_dim: int
    latent_interface_dim: int # this is the dim of the space between the task and robot modules
    anchors: jnp.ndarray

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        task_anchor_state = jnp.concatenate([x, self.anchors], axis=0)
        x = nn.BatchNorm()(task_anchor_state)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.normalize(x)
        x_task = x[0:-self.latent_interface_dim, :]
        x_anchor = x[-self.latent_interface_dim:, :]

        relative_interface = jnp.matmul(x_task, x_anchor.T)
        return relative_interface

class RobotModule(nn.Module):
    num_actions: int
    num_robot_inputs : int
    latent_interface_dim: int # this is the dim of the space between the task and robot modules
    hidden_dim: int
    @nn.compact
    def __call__(self, x: jnp.ndarray, robot_state: jnp.ndarray) -> jnp.ndarray:
        action_emb = nn.BatchNorm()(robot_state)
        action_emb = nn.Dense(self.hidden_dim - self.latent_interface_dim, kernel_init=default_init())(action_emb)
        x = jnp.concatenate([action_emb, x], axis=1)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        x = nn.relu(x)

        mean = nn.Dense(self.num_actions, kernel_init=default_init())(x)
        log_std = nn.Dense(self.num_actions, kernel_init=default_init())(x)

        log_std = jnp.clip(log_std, a_min=LOG_SIG_MIN, a_max=LOG_SIG_MAX)
        
        return mean, log_std
