"""Train and evaluate RL algorithms on OurSingleRLAviary.

Supported algorithms: PPO, SAC, TD3, DDPG (all from Stable-Baselines3).

Usage:
    python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ppo
    python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo sac
    python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo td3
    python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ddpg
"""

import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import str2bool, sync

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_ALGO = "ppo"
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_OBS = ObservationType("kin")
DEFAULT_ACT = ActionType("vel")
DEFAULT_TOTAL_TIMESTEPS = int(1e6)
DEFAULT_EVAL_FREQ = int(1e4)
DEFAULT_N_EVAL_EPISODES = 3
DEFAULT_VISUALIZE_COVERAGE = True

ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}

# ── Algorithm-specific hyper-parameters ──────────────────────────────
ALGO_KWARGS = {
    "ppo": dict(
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        n_epochs=5,
        clip_range=0.2,
    ),
    "sac": dict(
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=int(1e5),
        learning_starts=1000,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
    ),
    "td3": dict(
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=int(1e5),
        learning_starts=1000,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_delay=2,
    ),
    "ddpg": dict(
        batch_size=256,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=int(1e5),
        learning_starts=1000,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
    ),
}


def _make_action_noise(env, algo: str):
    """Create exploration noise for off-policy algorithms that benefit from it."""
    if algo in ("td3", "ddpg"):
        n_actions = env.action_space.shape[-1]
        return NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
    return None


# ── Main ─────────────────────────────────────────────────────────────
def run(
    algo: str = DEFAULT_ALGO,
    output_folder: str = DEFAULT_OUTPUT_FOLDER,
    gui: bool = DEFAULT_GUI,
    plot: bool = True,
    record_video: bool = DEFAULT_RECORD_VIDEO,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    eval_freq: int = DEFAULT_EVAL_FREQ,
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES,
    visualize_coverage: bool = DEFAULT_VISUALIZE_COVERAGE,
):
    algo = algo.lower()
    if algo not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from {list(ALGO_MAP)}")

    AlgoCls = ALGO_MAP[algo]
    tag = f"{algo}-{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}"
    save_dir = os.path.join(output_folder, tag)
    os.makedirs(save_dir, exist_ok=True)

    # ── Environments ──────────────────────────────────────────────
    env_kwargs = dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, gui=False, record=False)

    train_env = make_vec_env(OurSingleRLAviary, env_kwargs=env_kwargs, n_envs=1, seed=0)
    eval_env = OurSingleRLAviary(**env_kwargs)

    print(f"[INFO] Algorithm : {algo.upper()}")
    print(f"[INFO] Action    : {train_env.action_space}")
    print(f"[INFO] Obs       : {train_env.observation_space}")
    print(f"[INFO] Save dir  : {save_dir}")

    # ── Model ─────────────────────────────────────────────────────
    kwargs = dict(ALGO_KWARGS[algo])  # copy
    noise = _make_action_noise(train_env, algo)
    if noise is not None:
        kwargs["action_noise"] = noise

    model = AlgoCls(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        device="auto",
        tensorboard_log=os.path.join("runs", algo),
        **kwargs,
    )

    # ── Callbacks ─────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        verbose=1,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=int(eval_freq),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    # ── Train ─────────────────────────────────────────────────────
    model.learn(
        total_timesteps=int(total_timesteps),
        callback=eval_callback,
        log_interval=100,
    )
    model.save(os.path.join(save_dir, "final_model.zip"))
    print(f"\n[INFO] Training complete. Models saved to {save_dir}")

    # ── Plot learning curve ───────────────────────────────────────
    eval_file = os.path.join(save_dir, "evaluations.npz")
    if os.path.isfile(eval_file):
        with np.load(eval_file) as data:
            timesteps = data["timesteps"]
            results = data["results"].mean(axis=1)
            print("\n[DATA] evaluations.npz")
            for idx in range(timesteps.shape[0]):
                print(f"  step={timesteps[idx]:>8d}  reward={results[idx]:+.2f}")
            if plot:
                plt.figure(figsize=(10, 5))
                plt.plot(timesteps, results, marker="o", markersize=3, linewidth=1)
                plt.xlabel("Training Steps")
                plt.ylabel("Mean Episode Reward")
                plt.title(f"{algo.upper()} — OurSingleRLAviary")
                plt.grid(True, alpha=0.4)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=150)
                plt.show()

    # ── Evaluate best model ───────────────────────────────────────
    best_path = os.path.join(save_dir, "best_model.zip")
    model_path = best_path if os.path.isfile(best_path) else os.path.join(save_dir, "final_model.zip")
    model = AlgoCls.load(model_path)

    test_env_nogui = OurSingleRLAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=5)
    print(f"\n[RESULT] Mean reward = {mean_reward:.2f} ± {std_reward:.2f}\n")
    test_env_nogui.close()

    # ── GUI demo ──────────────────────────────────────────────────
    if gui:
        test_env = OurSingleRLAviary(
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            gui=True,
            record=record_video,
            visualize_coverage=visualize_coverage,
        )
        obs, info = test_env.reset()
        start = time.time()
        horizon = int((test_env.EPISODE_LEN_SEC + 2) * test_env.CTRL_FREQ) * 4
        for i in range(horizon):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            dr = info.get("drone_reward", reward)
            print(
                f"Step {i:04d}  reward={dr:+.4f}  "
                f"coverage={info['coverage_ratio']:.3f}  "
                f"captures={info['target_capture_count']}"
            )
            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated or truncated:
                obs, info = test_env.reset()
        test_env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SB3 RL training for OurSingleRLAviary")
    p.add_argument("--algo", default=DEFAULT_ALGO, type=str, choices=list(ALGO_MAP), help="RL algorithm")
    p.add_argument("--total_timesteps", default=DEFAULT_TOTAL_TIMESTEPS, type=int, help="Total env steps")
    p.add_argument("--eval_freq", default=DEFAULT_EVAL_FREQ, type=int, help="Evaluate every N steps")
    p.add_argument("--n_eval_episodes", default=DEFAULT_N_EVAL_EPISODES, type=int, help="Episodes per eval")
    p.add_argument("--gui", default=DEFAULT_GUI, type=str2bool, help="Show GUI demo after training")
    p.add_argument("--plot", default=True, type=str2bool, help="Show learning curve plot")
    p.add_argument("--record_video", default=DEFAULT_RECORD_VIDEO, type=str2bool, help="Record video")
    p.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str, help="Output folder")
    p.add_argument("--visualize_coverage", default=DEFAULT_VISUALIZE_COVERAGE, type=str2bool, help="Draw coverage rings")
    args = p.parse_args()
    run(**vars(args))
