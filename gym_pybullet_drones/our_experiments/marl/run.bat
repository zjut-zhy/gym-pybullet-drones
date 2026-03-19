@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PYTHON=D:\anaconda3\envs\drones\python.exe
set NUM_DRONES=4
set TOTAL_TIMESTEPS=1000000
set SEED=1
set GAMMA=0.99
set COMMON=--num-drones %NUM_DRONES% --total-timesteps %TOTAL_TIMESTEPS% --seed %SEED% --gamma %GAMMA% --save-model

echo.
echo ============================================================
echo   MARL Training - %NUM_DRONES% drones, %TOTAL_TIMESTEPS% steps, seed=%SEED%
echo ============================================================

echo.
echo [1/4] MAPPO ...
"%PYTHON%" -u "%SCRIPT_DIR%mappo.py" %COMMON% --learning-rate 3e-4 --num-steps 4096 --num-minibatches 32 --update-epochs 10 --clip-coef 0.2 --ent-coef 0.01 --vf-coef 0.5 --gae-lambda 0.95 --max-grad-norm 0.5 --target-kl 0.03
echo MAPPO exit code: %errorlevel%

echo.
echo [2/4] MASAC ...
"%PYTHON%" -u "%SCRIPT_DIR%masac.py" %COMMON% --policy-lr 3e-4 --q-lr 1e-3 --buffer-size 1000000 --batch-size 256 --learning-starts 5000 --tau 0.005 --autotune --alpha 0.2 --policy-frequency 2
echo MASAC exit code: %errorlevel%

echo.
echo [3/4] MADDPG ...
"%PYTHON%" -u "%SCRIPT_DIR%maddpg.py" %COMMON% --learning-rate 3e-4 --buffer-size 1000000 --batch-size 256 --learning-starts 25000 --exploration-noise 0.1 --tau 0.005 --policy-frequency 2
echo MADDPG exit code: %errorlevel%

echo.
echo [4/4] MATD3 ...
"%PYTHON%" -u "%SCRIPT_DIR%matd3.py" %COMMON% --learning-rate 3e-4 --buffer-size 1000000 --batch-size 256 --learning-starts 25000 --exploration-noise 0.1 --policy-noise 0.2 --noise-clip 0.5 --tau 0.005 --policy-frequency 2
echo MATD3 exit code: %errorlevel%

echo.
echo ============================================================
echo   Done! Models saved in runs/
echo   tensorboard: python -m tensorboard.main --logdir runs
echo ============================================================
pause
