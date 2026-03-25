@echo off
setlocal
set ENV=drones
set OUT1=results\go_explore
set OUT2=results\go_explore_phase2
set ITERS1=20000
set ENVS1=4
set ITERS2=10000
set ENVS2=4

call conda activate %ENV%
if errorlevel 1 (
    echo [ERROR] conda activate failed
    pause
    exit /b 1
)

echo ======================================================================
echo  Go-Explore Full Pipeline
echo ======================================================================

echo.
echo [1/4] Phase 1: Deterministic Exploration ...
python -m gym_pybullet_drones.our_experiments.go_explore.train --total_iterations %ITERS1% --n_envs %ENVS1% --output_dir %OUT1%
if errorlevel 1 (
    echo [ERROR] Phase 1 failed
    pause
    exit /b 1
)

echo.
echo [2/4] Bridge: Generate Demo ...
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo --archive_path %OUT1%\archive.json --output_path %OUT1%\best_demo.demo.pkl
if errorlevel 1 (
    echo [ERROR] Demo generation failed
    pause
    exit /b 1
)

echo.
echo [3/4] Phase 2: Robustification ...
python -m gym_pybullet_drones.our_experiments.go_explore.robustify --demo_path %OUT1%\best_demo.demo.pkl --total_iterations %ITERS2% --n_envs %ENVS2% --output_dir %OUT2%
if errorlevel 1 (
    echo [ERROR] Phase 2 failed
    pause
    exit /b 1
)

echo.
echo [4/4] Demo: GUI Visualization ...
python -m gym_pybullet_drones.our_experiments.go_explore.demo --model_path %OUT2%\model_final.pt --n_episodes 3

echo.
echo ======================================================================
echo  Done! Model saved at %OUT2%\model_final.pt
echo ======================================================================
pause
