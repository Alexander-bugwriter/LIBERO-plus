#!/usr/bin/env bash
set -eu          # 变量未定义/命令失败立即退出
export EVO_TEST_ROOT="/home/dell/code/ljt/logs/evo_da3_film_true_bellm_libero_plus/libero_spatial"
export EVO_STEP="80000"
export EVO_CKPT_TMPL="step_${EVO_STEP}/{filter_category}_total_h{horizon}_S0_test1"
export EVO_SERVER_URL="ws://127.0.0.1:8000"
# export EVO_TASK_SUITES="libero_spatial,libero_goal"   # 多个用逗号分隔
export EVO_TASK_SUITES="libero_spatial"   # 多个用逗号分隔
python -m evo_da2_clients.camera &&
python -m evo_da2_clients.robot &&
python -m evo_da2_clients.language &&
python -m evo_da2_clients.light &&
python -m evo_da2_clients.background &&
python -m evo_da2_clients.layout  &&
python -m evo_da2_clients.noise
 