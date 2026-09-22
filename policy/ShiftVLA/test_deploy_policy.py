"""CPU checks for the simulator/client/server observation and action contract."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY_DIR = Path(__file__).resolve().parent
bridge = load_module("shiftvla_bridge", POLICY_DIR / "deploy_policy.py")


def raw_observation(step=0):
    state = np.arange(14, dtype=np.float32)
    state[0] = step
    return {
        "joint_action": {
            "left_arm": state[:6],
            "left_gripper": state[6],
            "right_arm": state[7:13],
            "right_gripper": state[13],
            "vector": state,
        },
        "observation": {
            name: {"rgb": np.arange(36, dtype=np.uint8).reshape(3, 4, 3) + index}
            for index, name in enumerate(("head_camera", "left_camera", "right_camera"))
        },
    }


class FakeEnvironment:
    render_freq = 0
    eval_video_path = None
    step_lim = 3

    def __init__(self):
        self.executed = []

    def setup_demo(self, **kwargs):
        self.take_action_cnt = 0
        self.eval_success = False

    def close_env(self, **kwargs):
        pass

    def set_instruction(self, instruction):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction

    def get_obs(self):
        return raw_observation(self.take_action_cnt)

    def take_action(self, action, action_type):
        self.executed.append((action.copy(), action_type))
        self.take_action_cnt += 1


class ShiftVLAPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real rollout loop without importing Sapien or robot assets.
        modules = {
            "envs": types.SimpleNamespace(CONFIGS_PATH="unused/"),
            "envs.utils.create_actor": types.SimpleNamespace(UnStableError=type("UnStableError", (Exception,), {})),
            "generate_episode_instructions": types.SimpleNamespace(generate_episode_descriptions=mock.Mock()),
        }
        with mock.patch.dict(sys.modules, modules):
            cls.runner = load_module("shiftvla_test_runner", POLICY_DIR.parents[1] / "scripts/eval_policy_xpolicylab.py")

    def setUp(self):
        self.actions = np.arange(56, dtype=np.float32).reshape(4, 14)
        self.remote = mock.Mock()
        self.remote.infer.return_value = {"actions": self.actions}
        client_module = types.SimpleNamespace(WebsocketClientPolicy=mock.Mock(return_value=self.remote))
        with mock.patch.dict(sys.modules, {"openpi_client.websocket_client_policy": client_module}):
            self.policy = bridge.ShiftVLAPolicy(action_chunk_steps=2)

    def observation(self, instruction="move the block"):
        return self.runner.robotwin_obs_to_xpolicylab(raw_observation(), instruction=instruction)

    def test_three_cameras_and_raw_joint_order(self):
        source = raw_observation()
        encoded = bridge.encode_obs(self.observation())
        np.testing.assert_array_equal(encoded["state"], source["joint_action"]["vector"])
        for target, camera in (
            ("cam_high", "head_camera"),
            ("cam_left_wrist", "left_camera"),
            ("cam_right_wrist", "right_camera"),
        ):
            image = encoded["images"][target]
            self.assertEqual(image.dtype, np.uint8)
            np.testing.assert_array_equal(image, source["observation"][camera]["rgb"].transpose(2, 0, 1))
        self.assertEqual(encoded["prompt"], "move the block")

    def test_replanning_and_episode_reset_in_real_runner(self):
        environment = FakeEnvironment()
        task_args = {
            "task_name": "test_task",
            "policy_name": "ShiftVLA",
            "task_config": "demo_clean",
            "ckpt_setting": "test",
            "clear_cache_freq": 5,
            "render_freq": 0,
        }
        with mock.patch.object(self.runner, "build_instruction", side_effect=["first episode", "second episode"]):
            _, successes = self.runner.eval_remote_policy(
                "test_task", environment, task_args, self.policy, 100000,
                usr_args={"expert_check": False}, test_num=2,
            )
        self.assertEqual(successes, 0)
        self.assertEqual(self.remote.reset.call_count, 2)
        self.assertEqual(len(environment.executed), 6)
        self.assertEqual([call.args[0]["prompt"] for call in self.remote.infer.call_args_list],
                         ["first episode", "first episode", "second episode", "second episode"])
        self.assertEqual([call.args[0]["state"][0] for call in self.remote.infer.call_args_list], [0, 2, 0, 2])
        for (action, action_type), index in zip(environment.executed, [0, 1, 0, 0, 1, 0]):
            self.assertEqual(action_type, "qpos")
            np.testing.assert_array_equal(action, self.actions[index])
        self.policy.call("reset")
        with self.assertRaisesRegex(RuntimeError, "Update the observation"):
            self.policy.call("get_action")
        self.policy.close()
        self.remote._ws.close.assert_called_once()

    def test_invalid_dimensions_are_rejected(self):
        observation = self.observation()
        observation["state"]["left_arm_joint_state"] = np.zeros(7)
        with self.assertRaisesRegex(ValueError, "14-dimensional"):
            bridge.encode_obs(observation)
        self.policy.call("update_obs", self.observation())
        self.remote.infer.return_value = {"actions": np.zeros((2, 32))}
        with self.assertRaisesRegex(ValueError, "steps, 14"):
            self.policy.call("get_action")


if __name__ == "__main__":
    unittest.main()
