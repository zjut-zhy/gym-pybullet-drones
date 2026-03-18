# Multi-Agent DDPG (MADDPG) with parameter sharing for OurRLAviary.
# CleanRL-style single-file implementation.
#
# Usage:
#   python maddpg_continuous_action.py --num-drones 4 --total-timesteps 500000
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "MARL-Drones"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Environment
    num_drones: int = 2
    """number of drones in the swarm"""

    # Algorithm specific arguments
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Key order matching OurRLAviary._computeObs() return dict
OBS_KEYS = ["self_state", "teammate_state", "target_state", "obstacle_state"]


def flatten_dict_obs(obs: dict, keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(obs[k], dtype=np.float32).ravel() for k in keys])


def obs_dim_from_space(obs_space) -> int:
    return sum(int(np.prod(obs_space[k].shape)) for k in OBS_KEYS)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim):
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def add(self, obs, next_obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]).to(device),
            torch.as_tensor(self.next_obs[idx]).to(device),
            torch.as_tensor(self.actions[idx]).to(device),
            torch.as_tensor(self.rewards[idx]).to(device),
            torch.as_tensor(self.dones[idx]).to(device),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, config=vars(args), name=run_name, save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env = OurRLAviaryPZ(num_drones=args.num_drones, gui=False)
    agents_list = list(env.possible_agents)
    n_drones = len(agents_list)
    obs_space = env.observation_space(agents_list[0])
    act_space = env.action_space(agents_list[0])
    obs_keys = OBS_KEYS
    single_obs_dim = obs_dim_from_space(obs_space)
    single_act_dim = int(np.prod(act_space.shape))

    actor = Actor(single_obs_dim, single_act_dim).to(device)
    qf1 = QNetwork(single_obs_dim, single_act_dim).to(device)
    qf1_target = QNetwork(single_obs_dim, single_act_dim).to(device)
    target_actor = Actor(single_obs_dim, single_act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    rb = ReplayBuffer(args.buffer_size, single_obs_dim, single_act_dim)
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs_dict, _ = env.reset()
    ep_reward = 0.0

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        flat_obs_list = [flatten_dict_obs(obs_dict[ag], obs_keys) for ag in agents_list]
        flat_obs_t = torch.tensor(np.array(flat_obs_list), dtype=torch.float32).to(device)

        if global_step < args.learning_starts:
            act_np = np.array([act_space.sample() for _ in agents_list])
        else:
            with torch.no_grad():
                act_t = actor(flat_obs_t)
                act_t += torch.normal(0, args.exploration_noise, size=act_t.shape).to(device)
                act_np = act_t.cpu().numpy().clip(-1, 1)

        actions_dict = {agents_list[i]: act_np[i] for i in range(n_drones)}

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs_dict, rews, terms, truncs, infos = env.step(actions_dict)
        done = any(terms.get(ag, False) or truncs.get(ag, False) for ag in agents_list)
        ep_reward += sum(rews.get(ag, 0.0) for ag in agents_list)

        # store per-drone transitions
        next_flat_list = [flatten_dict_obs(next_obs_dict.get(ag, obs_dict[ag]), obs_keys) for ag in agents_list]
        for i, ag in enumerate(agents_list):
            rb.add(flat_obs_list[i], next_flat_list[i], act_np[i], rews.get(ag, 0.0), float(done))

        if done:
            writer.add_scalar("charts/episodic_return", ep_reward, global_step)
            print(f"global_step={global_step}, episodic_return={ep_reward:.2f}")
            obs_dict, _ = env.reset()
            ep_reward = 0.0
        else:
            obs_dict = next_obs_dict

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data_obs, data_next_obs, data_actions, data_rewards, data_dones = rb.sample(args.batch_size, device)

            with torch.no_grad():
                next_state_actions = target_actor(data_next_obs)
                qf1_next_target = qf1_target(data_next_obs, next_state_actions)
                next_q_value = data_rewards + (1 - data_dones) * args.gamma * qf1_next_target.view(-1)

            qf1_a_values = qf1(data_obs, data_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data_obs, actor(data_obs)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save((actor.state_dict(), qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")

    env.close()
    writer.close()
