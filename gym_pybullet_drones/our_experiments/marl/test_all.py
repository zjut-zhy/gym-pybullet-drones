"""Quick test for all 4 CleanRL-style MARL scripts."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import numpy as np
import torch

results = []

try:
    from gym_pybullet_drones.our_experiments.marl.mappo import Agent, flatten_dict_obs, obs_dim_from_space
    from gym_pybullet_drones.envs.OurRLAviary_PettingZoo import OurRLAviaryPZ

    env = OurRLAviaryPZ(num_drones=2, gui=False)
    agents_list = list(env.possible_agents)
    obs_space = env.observation_space(agents_list[0])
    act_space = env.action_space(agents_list[0])
    obs_keys = ["self_state", "teammate_state", "target_state", "obstacle_state"]
    obs_dim = obs_dim_from_space(obs_space)
    act_dim = int(np.prod(act_space.shape))

    agent = Agent(obs_dim, act_dim)
    obs_dict, _ = env.reset()
    flat = np.stack([flatten_dict_obs(obs_dict[ag], obs_keys) for ag in agents_list])
    flat_t = torch.tensor(flat, dtype=torch.float32)
    act, lp, ent, val = agent.get_action_and_value(flat_t)
    assert act.shape == (2, act_dim)
    results.append(f"MAPPO: OK (obs_dim={obs_dim}, act_dim={act_dim})")
except Exception as e:
    results.append(f"MAPPO: FAIL ({e})")

try:
    from gym_pybullet_drones.our_experiments.marl.masac import Actor as SacActor, SoftQNetwork
    actor_sac = SacActor(obs_dim, act_dim)
    qf = SoftQNetwork(obs_dim, act_dim)
    a, lp, m = actor_sac.get_action(flat_t)
    q = qf(flat_t, a)
    assert a.shape == (2, act_dim) and q.shape == (2, 1)
    results.append(f"MASAC: OK")
except Exception as e:
    results.append(f"MASAC: FAIL ({e})")

try:
    from gym_pybullet_drones.our_experiments.marl.maddpg import Actor as DdpgActor, QNetwork as DdpgQ
    actor_ddpg = DdpgActor(obs_dim, act_dim)
    qn = DdpgQ(obs_dim, act_dim)
    da = actor_ddpg(flat_t)
    dq = qn(flat_t, da)
    assert da.shape == (2, act_dim) and dq.shape == (2, 1)
    results.append(f"MADDPG: OK")
except Exception as e:
    results.append(f"MADDPG: FAIL ({e})")

try:
    from gym_pybullet_drones.our_experiments.marl.matd3 import Actor as Td3Actor, QNetwork as Td3Q
    actor_td3 = Td3Actor(obs_dim, act_dim)
    qf1_td3 = Td3Q(obs_dim, act_dim)
    qf2_td3 = Td3Q(obs_dim, act_dim)
    ta = actor_td3(flat_t)
    tq1 = qf1_td3(flat_t, ta)
    tq2 = qf2_td3(flat_t, ta)
    assert ta.shape == (2, act_dim) and tq1.shape == (2, 1) and tq2.shape == (2, 1)
    results.append(f"MATD3: OK (twin Q)")
except Exception as e:
    results.append(f"MATD3: FAIL ({e})")

try:
    obs_dict, _ = env.reset()
    for step in range(5):
        flat = np.stack([flatten_dict_obs(obs_dict[ag], obs_keys) for ag in agents_list])
        acts = agent.get_action_and_value(torch.tensor(flat, dtype=torch.float32))[0].detach().numpy().clip(-1,1)
        actions_dict = {agents_list[i]: acts[i] for i in range(len(agents_list))}
        obs_next, rews, terms, truncs, _ = env.step(actions_dict)
        done = any(terms.get(ag, False) or truncs.get(ag, False) for ag in agents_list)
        if done: obs_dict, _ = env.reset()
        else: obs_dict = obs_next
    results.append("ENV STEP: OK")
except Exception as e:
    results.append(f"ENV STEP: FAIL ({e})")

env.close()

print("=" * 50)
for r in results:
    print(r)
all_ok = all("OK" in r for r in results)
print("=" * 50)
print("ALL PASSED" if all_ok else "SOME FAILED")
