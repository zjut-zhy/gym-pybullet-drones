"""Snapshot save/restore end-to-end verification.

Tests:
1. get_snapshot() / restore_snapshot() on OurSingleRLAviary
2. Deterministic replay: same actions from same snapshot produce same results
3. OurSingleRLAviary inherits snapshot methods from OurRLAviary

Usage
-----
    python tests/test_snapshot.py
"""

import sys
import numpy as np

from gym_pybullet_drones.envs.OurSingleRLAviary import OurSingleRLAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType


def _obs_equal(obs_a, obs_b, atol=1e-6) -> bool:
    """Check two observation dicts are equal."""
    if set(obs_a.keys()) != set(obs_b.keys()):
        return False
    for k in obs_a:
        if not np.allclose(obs_a[k], obs_b[k], atol=atol):
            return False
    return True


def test_snapshot_roundtrip():
    """Snapshot -> advance -> restore -> obs must match snapshot point."""
    print("\n[TEST 1] Snapshot round-trip ...")
    env = OurSingleRLAviary(
        obs=ObservationType.KIN, act=ActionType.VEL,
        gui=False, record=False,
        arena_size_xy_m=10.0, target_count=6, obstacle_count=3,
        ctrl_freq=30, max_episode_len_sec=30.0, environment_seed=123,
    )

    obs_reset, _ = env.reset()
    rng = np.random.RandomState(42)

    # Run 20 steps
    for _ in range(20):
        action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    # -- take snapshot --
    snap = env.get_snapshot()
    obs_at_snap = env._computeObs()

    # Run 30 more steps (mutate state)
    for _ in range(30):
        action = rng.uniform(-1, 1, size=(2,)).astype(np.float32)
        env.step(action)

    obs_after_mutate = env._computeObs()
    assert not _obs_equal(obs_at_snap, obs_after_mutate), \
        "State should have changed after 30 steps"

    # -- restore snapshot --
    env.restore_snapshot(snap)
    obs_restored = env._computeObs()

    assert _obs_equal(obs_at_snap, obs_restored), \
        "Restored obs does not match snapshot obs!"

    print("  PASS: obs after restore matches snapshot point")
    env.close()


def test_deterministic_replay():
    """Same actions from same snapshot must produce identical trajectories."""
    print("\n[TEST 2] Deterministic replay from snapshot ...")
    env = OurSingleRLAviary(
        obs=ObservationType.KIN, act=ActionType.VEL,
        gui=False, record=False,
        arena_size_xy_m=10.0, target_count=6, obstacle_count=3,
        ctrl_freq=30, max_episode_len_sec=30.0, environment_seed=456,
    )

    obs, _ = env.reset()
    rng_warmup = np.random.RandomState(99)

    # Warm up 15 steps
    for _ in range(15):
        action = rng_warmup.uniform(-1, 1, size=(2,)).astype(np.float32)
        obs, rew, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    snap = env.get_snapshot()

    # Generate a fixed action sequence
    actions_to_test = [np.array([0.5, -0.3], dtype=np.float32),
                       np.array([-0.8, 0.1], dtype=np.float32),
                       np.array([0.2, 0.7], dtype=np.float32),
                       np.array([-0.1, -0.9], dtype=np.float32),
                       np.array([0.9, 0.4], dtype=np.float32)]

    # -- first run --
    env.restore_snapshot(snap)
    trajectory_a = []
    for act in actions_to_test:
        obs, rew, term, trunc, info = env.step(act)
        trajectory_a.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "reward": rew,
            "terminated": term,
            "truncated": trunc,
        })

    # -- scramble state --
    for _ in range(20):
        env.step(np.array([0.0, 0.0], dtype=np.float32))

    # -- second run from same snapshot --
    env.restore_snapshot(snap)
    trajectory_b = []
    for act in actions_to_test:
        obs, rew, term, trunc, info = env.step(act)
        trajectory_b.append({
            "obs": {k: v.copy() for k, v in obs.items()},
            "reward": rew,
            "terminated": term,
            "truncated": trunc,
        })

    # Compare
    for i, (a, b) in enumerate(zip(trajectory_a, trajectory_b)):
        assert _obs_equal(a["obs"], b["obs"]), \
            f"Step {i}: obs mismatch"
        assert abs(a["reward"] - b["reward"]) < 1e-6, \
            f"Step {i}: reward mismatch ({a['reward']} vs {b['reward']})"
        assert a["terminated"] == b["terminated"], \
            f"Step {i}: terminated mismatch"
        assert a["truncated"] == b["truncated"], \
            f"Step {i}: truncated mismatch"

    print(f"  PASS: {len(actions_to_test)} steps produce identical results from same snapshot")
    env.close()


def test_inheritance():
    """OurSingleRLAviary should inherit get_snapshot / restore_snapshot."""
    print("\n[TEST 3] Inheritance check ...")
    env = OurSingleRLAviary(
        obs=ObservationType.KIN, act=ActionType.VEL,
        gui=False, record=False,
        arena_size_xy_m=10.0, target_count=4, obstacle_count=2,
        ctrl_freq=30, max_episode_len_sec=10.0, environment_seed=789,
    )
    env.reset()

    assert hasattr(env, "get_snapshot"), "Missing get_snapshot method"
    assert hasattr(env, "restore_snapshot"), "Missing restore_snapshot method"
    assert callable(env.get_snapshot), "get_snapshot not callable"
    assert callable(env.restore_snapshot), "restore_snapshot not callable"

    snap = env.get_snapshot()
    assert isinstance(snap, dict), f"get_snapshot returned {type(snap)}, expected dict"

    expected_keys = {"pos", "quat", "rpy", "vel", "ang_v", "step_counter",
                     "rng_state", "target_positions", "obstacle_positions_xy"}
    missing = expected_keys - set(snap.keys())
    assert not missing, f"Snapshot missing keys: {missing}"

    env.restore_snapshot(snap)
    print("  PASS: OurSingleRLAviary inherits snapshot methods")
    env.close()


if __name__ == "__main__":
    try:
        test_snapshot_roundtrip()
        test_deterministic_replay()
        test_inheritance()
        print("\n" + "=" * 50)
        print("ALL SNAPSHOT TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n  FAIL: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
