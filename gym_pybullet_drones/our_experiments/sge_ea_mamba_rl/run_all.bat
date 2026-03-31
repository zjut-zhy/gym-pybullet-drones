@echo off
REM SEAM-RL: Full pipeline (Phase 1 + Phase 2)
REM
REM Phase 1: SAC-guided Go-Explore
REM Phase 2: EA-Mamba RL + SIL + Backward Curriculum

echo ============================================================
echo  SEAM-RL: Phase 1 -- SAC-Guided Go-Explore
echo ============================================================

python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.train ^
    --total_iterations 20000 ^
    --explore_steps 300 ^
    --output_dir results/sge_ea_mamba_rl_phase1

echo.
echo Phase 1 complete. Generating demo trajectories...
echo.

REM Generate demos from archive (reuse go_explore's gen_demo)
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo ^
    --archive_path results/sge_ea_mamba_rl_phase1/archive.json ^
    --output_path results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl ^
    --top_k 5

echo.
echo ============================================================
echo  SEAM-RL: Phase 2 -- EA-Mamba RL + SIL + Backward Curriculum
echo ============================================================

python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.robustify ^
    --demo_path results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl ^
    --total_timesteps 1000000 ^
    --output_dir results/sge_ea_mamba_rl

echo.
echo ============================================================
echo  SEAM-RL pipeline complete!
echo ============================================================
pause
