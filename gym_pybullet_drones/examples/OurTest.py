"""Train and evaluate PPO on OurRLAviary.

This follows the structure of examples/learn.py, but is adapted for:
- OurRLAviary
- Dict observations
- PPO MultiInputPolicy

Run:

    python gym_pybullet_drones/examples/OurTest.py

"""

import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# from gym_pybullet_drones.envs.OurRLAviary import OurRLAviary
from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary as OurRLAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool, sync


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("vel")
DEFAULT_AGENTS = 1
DEFAULT_TOTAL_TIMESTEPS = int(1e4)
DEFAULT_EVAL_FREQ = 1e4
DEFAULT_VISUALIZE_COVERAGE = True


def _extract_log_state(obs_dict: dict, drone_idx: int, action: np.ndarray) -> np.ndarray:
    """Build a 20-D state-like vector for the existing logger from Dict observations."""
    ss = obs_dict["self_state"]
    # Single-agent env returns shape (dim,); multi-agent returns (N, dim)
    self_state = ss if ss.ndim == 1 else ss[drone_idx]
    pos = self_state[0:3]
    rpy = np.zeros(3, dtype=np.float32)
    vel = self_state[3:6]
    act = np.asarray(action).reshape(-1) if np.asarray(action).ndim == 1 else np.asarray(action[drone_idx]).reshape(-1)
    if act.shape[0] < 4:
        act = np.pad(act, (0, 4 - act.shape[0]))
    else:
        act = act[:4]
    return np.hstack([pos, np.zeros(4), rpy, vel, np.zeros(3), act]).astype(np.float32)


def run(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    gui=DEFAULT_GUI,
    plot=True,
    colab=DEFAULT_COLAB,
    record_video=DEFAULT_RECORD_VIDEO,
    local=True,
    num_drones=DEFAULT_AGENTS,
    total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
    eval_freq=DEFAULT_EVAL_FREQ,
    visualize_coverage=DEFAULT_VISUALIZE_COVERAGE,
):
    filename = os.path.join(output_folder, "save-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename, exist_ok=True)

    env_kwargs = dict(
        num_drones=num_drones,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        gui=False,
        record=False,
    )

    train_env = make_vec_env(
        OurRLAviary,
        env_kwargs=env_kwargs,
        n_envs=1,
        seed=0,
    )
    eval_env = OurRLAviary(**env_kwargs)

    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)

    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        device="cuda",
    )

    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=filename,
        log_path=filename,
        eval_freq=int(eval_freq),
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=int(total_timesteps) if local else int(1e3),
        callback=eval_callback,
        log_interval=100,
    )

    model.save(os.path.join(filename, "final_model.zip"))
    print(filename)

    eval_file = os.path.join(filename, "evaluations.npz")
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps = data["timesteps"]
            results = data["results"][:, 0]
            print("Data from evaluations.npz")
            for idx in range(timesteps.shape[0]):
                print(f"{timesteps[idx]},{results[idx]}")
            if local and plot:
                plt.plot(timesteps, results, marker="o", linestyle="-", markersize=4)
                plt.xlabel("Training Steps")
                plt.ylabel("Episode Reward")
                plt.grid(True, alpha=0.6)
                plt.show()

    best_model_path = os.path.join(filename, "best_model.zip")
    model_path = best_model_path if os.path.isfile(best_model_path) else os.path.join(filename, "final_model.zip")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No trained model found under {filename}")

    model = PPO.load(model_path)
    test_env_nogui = OurRLAviary(num_drones=num_drones, obs=DEFAULT_OBS, act=DEFAULT_ACT)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=5)
    print("\n\nMean reward", mean_reward, "+-", std_reward, "\n")

    test_env = OurRLAviary(
        num_drones=num_drones,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        gui=gui,
        record=record_video,
        visualize_coverage=visualize_coverage,
    )

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    obs, info = test_env.reset()
    start = time.time()
    horizon = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ) * 4
    for i in range(horizon):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        drone_r = info.get("drone_rewards", info.get("drone_reward", np.zeros(num_drones)))
        if np.isscalar(drone_r) or (isinstance(drone_r, np.ndarray) and drone_r.ndim == 0):
            drone_r = np.array([float(drone_r)])
        reward_str = " ".join(f"d{k}={drone_r[k]:+.4f}" for k in range(min(num_drones, len(drone_r))))
        print(
            f"Step {i:04d} reward=[{reward_str}] sum={reward:.4f} terminated={terminated} truncated={truncated} "
            f"coverage={info['coverage_ratio']:.3f} captures={info['target_capture_count']}"
        )

        for drone_idx in range(num_drones):
            logger.log(
                drone=drone_idx,
                timestamp=i / test_env.CTRL_FREQ,
                state=_extract_log_state(obs, drone_idx, action),
                control=np.zeros(12),
            )

        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated or truncated:
            obs, info = test_env.reset()

    test_env.close()
    test_env_nogui.close()

    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training script for OurRLAviary")
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, help="Whether to use PyBullet GUI", metavar="")
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VIDEO, type=str2bool, help="Whether to record a video", metavar="")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, help="Folder where to save logs", metavar="")
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=bool, help="Whether example is being run by a notebook", metavar="")
    parser.add_argument("--num_drones", default=DEFAULT_AGENTS, type=int, help="Number of cooperative UAVs", metavar="")
    parser.add_argument("--total_timesteps", default=DEFAULT_TOTAL_TIMESTEPS, type=int, help="PPO training steps", metavar="")
    parser.add_argument("--eval_freq", default=DEFAULT_EVAL_FREQ, type=int, help="Evaluation frequency", metavar="")
    parser.add_argument("--plot", default=True, type=str2bool, help="Whether to show plots", metavar="")
    parser.add_argument("--local", default=True, type=str2bool, help="Use full training length or short debug training", metavar="")
    parser.add_argument("--visualize_coverage", default=DEFAULT_VISUALIZE_COVERAGE, type=str2bool, help="Whether to draw observation/threat/coverage circles in GUI", metavar="")
    ARGS = parser.parse_args()
    run(**vars(ARGS))