@echo off
REM SGE-EA (ablation): Full pipeline (Phase 1 + Phase 2)
REM
REM Ablation: Entity Attention only, NO Mamba temporal modeling.

echo ============================================================
echo  SGE-EA (ablation) Phase 1 -- SAC-Guided Go-Explore
echo ============================================================

python -m gym_pybullet_drones.our_experiments.sge_ea_rl.train ^
    --total_iterations 20000 ^
    --explore_steps 300 ^
    --output_dir results/sge_ea_rl_phase1

echo.
echo Phase 1 complete. Generating demo trajectories...
echo.

python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo ^
    --archive_path results/sge_ea_rl_phase1/archive.json ^
    --output_path results/sge_ea_rl_phase1/demos_best.demo.pkl ^
    --top_k 5

echo.
echo ============================================================
echo  SGE-EA (ablation) Phase 2 -- EA RL + SIL + Backward Curriculum
echo ============================================================

python -m gym_pybullet_drones.our_experiments.sge_ea_rl.robustify ^
    --demo_path results/sge_ea_rl_phase1/demos_best.demo.pkl ^
    --total_timesteps 1000000 ^
    --output_dir results/sge_ea_rl

echo.
echo ============================================================
echo  SGE-EA (ablation) pipeline complete!
echo ============================================================
pause
