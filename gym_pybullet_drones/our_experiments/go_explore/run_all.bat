@echo off
REM ======================================================================
REM  Go-Explore 一键训练脚本 (Phase 1 → Bridge → Phase 2 → Demo)
REM  使用方法: 双击运行 或 在命令行中执行 run_all.bat
REM ======================================================================

setlocal

REM --- 配置 ---
set CONDA_ENV=drones
set OUTPUT_DIR=results\go_explore
set PHASE2_DIR=results\go_explore_phase2

set PHASE1_ITERS=5000
set PHASE1_ENVS=4
set PHASE2_ITERS=3000
set PHASE2_ENVS=4
set SEED=42

REM --- 激活 conda ---
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] 无法激活 conda 环境 %CONDA_ENV%
    pause
    exit /b 1
)

echo ======================================================================
echo  Go-Explore 全流程训练
echo  Phase 1: %PHASE1_ITERS% iters, %PHASE1_ENVS% envs
echo  Phase 2: %PHASE2_ITERS% iters, %PHASE2_ENVS% envs
echo  Seed: %SEED%
echo ======================================================================

REM ====================== Phase 1: 探索 ======================
echo.
echo [1/4] Phase 1: 确定性探索与树状归档 ...
echo.

python -m gym_pybullet_drones.our_experiments.go_explore.train ^
    --total_iterations %PHASE1_ITERS% ^
    --n_envs %PHASE1_ENVS% ^
    --seed %SEED% ^
    --output_dir %OUTPUT_DIR%

if errorlevel 1 (
    echo [ERROR] Phase 1 失败
    pause
    exit /b 1
)

REM ====================== Bridge: Demo 生成 ======================
echo.
echo [2/4] Bridge: 树溯源 + 确定性回放生成 Demo ...
echo.

python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo ^
    --archive_path %OUTPUT_DIR%\archive.json ^
    --output_path %OUTPUT_DIR%\best_demo.demo.pkl

if errorlevel 1 (
    echo [ERROR] Demo 生成失败
    pause
    exit /b 1
)

REM ====================== Phase 2: 鲁棒化训练 ======================
echo.
echo [3/4] Phase 2: 后向课程 + PPO + SIL 鲁棒化训练 ...
echo.

python -m gym_pybullet_drones.our_experiments.go_explore.robustify ^
    --demo_path %OUTPUT_DIR%\best_demo.demo.pkl ^
    --total_iterations %PHASE2_ITERS% ^
    --n_envs %PHASE2_ENVS% ^
    --seed %SEED% ^
    --output_dir %PHASE2_DIR%

if errorlevel 1 (
    echo [ERROR] Phase 2 失败
    pause
    exit /b 1
)

REM ====================== Demo: 模型演示 ======================
echo.
echo [4/4] 加载最终模型进行 GUI 演示 ...
echo.

python -m gym_pybullet_drones.our_experiments.go_explore.demo ^
    --model_path %PHASE2_DIR%\model_final.pt ^
    --n_episodes 3

echo.
echo ======================================================================
echo  训练与演示全部完成!
echo  模型:  %PHASE2_DIR%\model_final.pt
echo  归档:  %OUTPUT_DIR%\archive.json
echo  Demo:  %OUTPUT_DIR%\best_demo.demo.pkl
echo ======================================================================

pause
