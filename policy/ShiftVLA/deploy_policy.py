"""Connect the RoboTwin evaluation runner to ShiftVLA's openpi policy server."""

from pathlib import Path
import sys

import numpy as np


OPENPI_CLIENT_SRC = Path(__file__).resolve().parents[3] / "openpi/packages/openpi-client/src"
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))


def encode_obs(observation):
    """Convert the runner's observation to the training policy's Aloha input."""
    state = observation["state"]
    joint_state = np.concatenate(
        [
            np.asarray(state[name], dtype=np.float32).reshape(-1)
            for name in (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        ]
    )
    if joint_state.shape != (14,):
        raise ValueError(f"ShiftVLA expects a 14-dimensional Aloha state, got {joint_state.shape}")

    # RoboTwin emits RGB HWC; AlohaInputs decodes RGB CHW on the server.
    # Joint signs, grippers, normalization and delta actions also belong to the
    # trained server transforms and must not be applied a second time here.
    return {
        "state": joint_state,
        "images": {
            target: np.ascontiguousarray(np.moveaxis(observation["vision"][source]["color"], -1, 0))
            for target, source in (
                ("cam_high", "cam_head"),
                ("cam_left_wrist", "cam_left_wrist"),
                ("cam_right_wrist", "cam_right_wrist"),
            )
        },
        "prompt": observation["instruction"],
    }


class ShiftVLAPolicy:
    """Expose the runner's reset/update_obs/get_action API over openpi WebSocket."""

    _robotwin_protocol = "openpi"

    def __init__(self, host="localhost", port=9000, action_chunk_steps=50):
        if action_chunk_steps <= 0:
            raise ValueError("action_chunk_steps must be positive")

        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        self.policy = WebsocketClientPolicy(host=host, port=int(port))
        self.action_chunk_steps = int(action_chunk_steps)
        self.observation = None

    def call(self, func_name, obs=None):
        if func_name == "reset":
            # The openpi policy server is stateless. Clear this episode's prompt
            # and observation locally; reset() does not send an inference RPC.
            self.observation = None
            self.policy.reset()
            return None
        if func_name == "update_obs":
            self.observation = encode_obs(obs)
            return None
        if func_name == "get_action":
            if self.observation is None:
                raise RuntimeError("Update the observation before requesting actions")
            actions = np.asarray(self.policy.infer(self.observation)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 14 or actions.shape[0] == 0:
                raise ValueError(f"ShiftVLA expects a nonempty (steps, 14) action chunk, got {actions.shape}")
            # The runner executes this prefix, stops on success/time limit, then
            # observes again before requesting a fresh chunk.
            return actions[: self.action_chunk_steps]
        raise ValueError(f"Unsupported policy call: {func_name}")

    def close(self):
        # The bundled openpi client does not expose a public close method.
        self.policy._ws.close()


def get_model(usr_args):
    if usr_args.get("action_type", "joint") not in {"joint", "qpos"}:
        raise ValueError("ShiftVLA RoboTwin checkpoints require joint actions")
    return ShiftVLAPolicy(
        host=usr_args.get("host", "localhost"),
        port=usr_args.get("port", 9000),
        action_chunk_steps=int(usr_args.get("action_chunk_steps", 50)),
    )
