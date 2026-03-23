@echo off
REM Run all SB3 algorithms sequentially on OurSingleRLAviary
REM Usage:  run_all.bat
REM         run_all.bat --total_timesteps 500000

set EXTRA_ARGS=%*

echo ============================================================
echo  [1/4] PPO
echo ============================================================
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ppo --gui False --plot False %EXTRA_ARGS%

echo ============================================================
echo  [2/4] SAC
echo ============================================================
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo sac --gui False --plot False %EXTRA_ARGS%

echo ============================================================
echo  [3/4] TD3
echo ============================================================
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo td3 --gui False --plot False %EXTRA_ARGS%

echo ============================================================
echo  [4/4] DDPG
echo ============================================================
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ddpg --gui False --plot False %EXTRA_ARGS%

echo ============================================================
echo  All training runs complete!
echo ============================================================
pause
