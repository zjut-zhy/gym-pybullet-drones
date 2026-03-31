@echo off
REM ======================================================================
REM  SGE-MambaRL Full Pipeline
REM  Phase 1: SAC-guided Go-Explore  ->  Phase 2: Mamba RL
REM ======================================================================
setlocal
set ENV=drones
set OUT1=results\sge_mambarl_phase1
set OUT2=results\sge_mambarl
set ITERS1=20000
set TIMESTEPS2=1000000

call conda activate %ENV%
if errorlevel 1 (
    echo [ERROR] conda activate failed
    pause
    exit /b 1
)

echo ======================================================================
echo  SGE-MambaRL Full Pipeline
echo ======================================================================

echo.
echo [1/4] Phase 1: SAC-Guided Go-Explore ...
python -m gym_pybullet_drones.our_experiments.sge_mambarl.train --total_iterations %ITERS1% --output_dir %OUT1%
if errorlevel 1 (
    echo [ERROR] Phase 1 failed
    pause
    exit /b 1
)

echo.
echo [2/4] Bridge: Generate Demo from Archive ...
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo --archive_path %OUT1%\archive.json --output_path %OUT1%\demos_best.demo.pkl
if errorlevel 1 (
    echo [ERROR] Demo generation failed
    pause
    exit /b 1
)

echo.
echo [3/4] Phase 2: Mamba RL + SIL + Backward Curriculum ...
python -m gym_pybullet_drones.our_experiments.sge_mambarl.robustify --demo_path %OUT1%\demos_best.demo.pkl --total_timesteps %TIMESTEPS2% --output_dir %OUT2%
if errorlevel 1 (
    echo [ERROR] Phase 2 failed
    pause
    exit /b 1
)

echo.
echo [4/4] Demo: GUI Visualization ...
python -m gym_pybullet_drones.our_experiments.sge_mambarl.demo --model_path %OUT2%\best_model.zip --n_episodes 3

echo.
echo ======================================================================
echo  Done! Model saved at %OUT2%\best_model.zip
echo ======================================================================
pause
