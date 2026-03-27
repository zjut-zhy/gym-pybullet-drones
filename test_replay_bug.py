"""Inline verification: after each archive update, replay every NEW cell's
   full_action_sequence to verify it reaches the same state as the snapshot."""

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gym_pybullet_drones.our_experiments.go_explore.archive import Archive
import numpy as np

def make_env():
    return OurSingleRLAviary(
        obs=ObservationType.KIN, act=ActionType.VEL, gui=False, record=False,
        arena_size_xy_m=10.0, target_count=18, obstacle_count=6, ctrl_freq=30,
        max_episode_len_sec=60.0, environment_seed=42)

rng = np.random.RandomState(42)
envs = [make_env() for _ in range(2)]
verify_env = make_env()  # separate env for verification replays
archive = Archive(cell_size=0.5, arena_half=5.0, env_seed=42)
archive.seed(42)

errors = 0
total_verified = 0

for iteration in range(1, 201):
    for env in envs:
        target_cell = archive.select() if len(archive) > 0 else None
        source_cell = None
        if target_cell is not None and target_cell.snapshot is not None:
            env.restore_snapshot(target_cell.snapshot)
            obs = env._computeObs()
            source_cell = target_cell
        else:
            obs, info = env.reset()

        all_obs, all_snap, all_rew, all_act, all_ncap = [], [], [], [], []
        for _ in range(300):
            action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
            obs, rew, term, trunc, info = env.step(action)
            n_cap = int(info.get("target_capture_count", 0))
            if term or trunc:
                if n_cap >= 18:
                    all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
                    all_snap.append(env.get_snapshot())
                    all_rew.append(float(rew))
                    all_act.append(np.array(action, copy=True))
                    all_ncap.append(n_cap)
                break
            all_obs.append({k: np.array(v, copy=True) for k, v in obs.items()})
            all_snap.append(env.get_snapshot())
            all_rew.append(float(rew))
            all_act.append(np.array(action, copy=True))
            all_ncap.append(n_cap)

        before = set(archive.cells.keys())
        archive.update(all_obs, all_snap, all_rew, all_act,
                       trajectory_n_captured=all_ncap, source_cell=source_cell)
        after = set(archive.cells.keys())

        # Verify each NEW cell
        for new_key in (after - before):
            cell = archive.cells[new_key]
            seq = cell.full_action_sequence
            if seq is None or len(seq) == 0:
                continue
            total_verified += 1
            # Replay
            verify_obs, _ = verify_env.reset()
            failed = False
            for i, a in enumerate(seq):
                verify_obs, r, t, tr, info = verify_env.step(a)
                if t or tr:
                    errors += 1
                    print("ERROR iter=%d: cell %s seq_len=%d FAILED at step %d/%d (term=%s trunc=%s)"
                          % (iteration, new_key, len(seq), i+1, len(seq), t, tr))
                    failed = True
                    break
            if not failed:
                # Check position matches
                replay_pos = verify_env.pos[0].copy()
                verify_env.restore_snapshot(cell.snapshot)
                snap_pos = verify_env.pos[0].copy()
                if not np.allclose(replay_pos, snap_pos, atol=1e-6):
                    errors += 1
                    print("ERROR iter=%d: cell %s POSITION MISMATCH replay=%s snap=%s"
                          % (iteration, new_key, replay_pos[:2], snap_pos[:2]))

    if iteration % 50 == 0:
        best = archive.get_best_cell()
        bseq = len(best.full_action_sequence) if best.full_action_sequence else 0
        print("iter=%d: cells=%d, best=%s seq=%d, verified=%d, errors=%d"
              % (iteration, len(archive), best.key, bseq, total_verified, errors))

print("\nDone. Total verified: %d, Total errors: %d" % (total_verified, errors))
verify_env.close()
for e in envs:
    e.close()
