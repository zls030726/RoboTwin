import os
import sys
from pathlib import Path

import numpy as np


# RoboTwin runs this module from /workspace/VLA/VLDA/RoboTwin, while the VLDA model
# and policy factory live in /workspace/VLA/VLDA/models. Add the repository root
# explicitly so `models.*` resolves to our VLDA implementation.
VLDA_ROOT = Path(__file__).resolve().parents[3]
OPENPI_SRC = VLDA_ROOT / "openpi" / "src"
for import_path in (VLDA_ROOT, OPENPI_SRC):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from models import config as _config
from models import policy_config as _policy_config


class VLDAPolicy:
    def __init__(
        self,
        train_config_name: str,
        checkpoint_dir: str | None = None,
        exp_name: str | None = None,
        ckpt_setting: int | str | None = None,
        action_chunk_steps: int | None = None,
        pytorch_device: str | None = None,
    ):
        self.train_config_name = train_config_name
        self.action_chunk_steps = action_chunk_steps
        self.observation_window = None
        self.instruction = None

        train_config = _config.get_config(train_config_name)
        ckpt_dir = self._resolve_checkpoint_dir(
            train_config,
            checkpoint_dir=checkpoint_dir,
            exp_name=exp_name,
            ckpt_setting=ckpt_setting,
        )

        self.policy = _policy_config.create_trained_policy(
            train_config,
            ckpt_dir,
            pytorch_device=pytorch_device,
        )
        print(f"Loaded VLDA policy from {ckpt_dir}")

    def _resolve_checkpoint_dir(self, train_config, *, checkpoint_dir, exp_name, ckpt_setting):
        if checkpoint_dir:
            return checkpoint_dir

        if exp_name is None or ckpt_setting is None:
            raise ValueError(
                "Either provide checkpoint_dir, or provide both exp_name/model_name "
                "and ckpt_setting in deploy_policy.yml or eval.sh overrides."
            )

        # Training saves checkpoints as:
        #   <checkpoint_base_dir>/<config_name>/<exp_name>/2/<ckpt_setting>
        # Stage 2 is the inference checkpoint for VLDA.
        return os.path.join(
            train_config.checkpoint_base_dir,
            train_config.name,
            str(exp_name),
            "2",
            str(ckpt_setting),
        )

    def set_language(self, instruction: str):
        self.instruction = instruction
        print(f"successfully set instruction: {instruction}")

    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left = img_arr

        # openpi AlohaInputs expects channel-first uint8 images; it converts
        # them back to HWC internally before resizing/tokenization.
        img_front = np.transpose(np.asarray(img_front), (2, 0, 1))
        img_right = np.transpose(np.asarray(img_right), (2, 0, 1))
        img_left = np.transpose(np.asarray(img_left), (2, 0, 1))

        self.observation_window = {
            "state": np.asarray(state),
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        if self.observation_window is None:
            raise RuntimeError("update_observation_window first!")
        actions = self.policy.infer(self.observation_window)["actions"]
        if self.action_chunk_steps is not None:
            actions = actions[: self.action_chunk_steps]
        return actions

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language instruction")


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name = usr_args.get("train_config_name") or "vlda_robotwin"
    checkpoint_dir = usr_args.get("checkpoint_dir")
    exp_name = usr_args.get("exp_name") or usr_args.get("model_name")
    ckpt_setting = usr_args.get("ckpt_setting")
    action_chunk_steps = usr_args.get("action_chunk_steps") or usr_args.get("pi0_step")
    pytorch_device = usr_args.get("pytorch_device")

    if action_chunk_steps is not None:
        action_chunk_steps = int(action_chunk_steps)

    return VLDAPolicy(
        train_config_name=train_config_name,
        checkpoint_dir=checkpoint_dir,
        exp_name=exp_name,
        ckpt_setting=ckpt_setting,
        action_chunk_steps=action_chunk_steps,
        pytorch_device=pytorch_device,
    )


def eval(TASK_ENV, model, observation):
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    model.reset_obsrvationwindows()
