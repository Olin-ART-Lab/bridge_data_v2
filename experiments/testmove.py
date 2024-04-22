import time

from absl import app, flags, logging

import numpy as np

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
FLAGS = flags.FLAGS
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist(
    "goal_eep", [0.3, 0.0, 0.15], "Goal position"
)  # not used by lc
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/D435/color/image_raw", "flip": True}]
FIXED_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}
def main(_):
    initial_eep = [float(e) for e in FLAGS.initial_eep]
    start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=128)
    widowx_client.reset()
    time.sleep(2.5)
    move_status = widowx_client.move(eep, duration=1.5)
    # widowx_client.step_action(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), blocking=True)

if __name__ == "__main__":
    app.run(main)