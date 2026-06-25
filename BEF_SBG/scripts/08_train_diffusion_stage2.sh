#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

mkdir -p "${DIFF_STAGE2_LOG_DIR}"

cd "${BGDIFF_ROOT}"

STAGE1_CKPT="${DIFF_STAGE1_LOG_DIR}/checkpoints/last.ckpt"

export PROJECT_DIR="${BGDIFF_ROOT}"
export PL_GLOBAL_SEED=0
export PYTHONHASHSEED=0

export RESUME_PATH="${RESUME_PATH:-${STAGE1_CKPT}}"
export PROMPT_JSON="${BEF_PROMPT_JSON}"

export LOG_ROOT="${DIFF_LOG_ROOT}"
export EXP_NAME="${DIFF_STAGE2_EXP}"

export NUM_GPUS=1
export PER_GPU_BATCH=1
export ACCUM=8
export LR=3e-6
export MAX_STEPS=20000
export LOGGER_FREQ=2000
export IMG_SIZE=384
export EMPTY_PROMPT_PROB=0.0
export PRECISION=16-mixed
export NUM_WORKERS=0

export SD_LOCKED=0
export ONLY_MID_CONTROL=0
export TRAIN_CONTROLNET=0
export TRAIN_UNET_DECODER=1

export BOUNDARY_PRIOR_MODE=external

export ENABLE_SOFT_BOUNDARY_PRIOR=1
export BOUNDARY_PRIOR_TAU=4.0
export BOUNDARY_PRIOR_RADIUS=12
export BOUNDARY_DILATE_KERNEL=3

export ENABLE_PROGRESSIVE_BOUNDARY_GUIDANCE=1
export BOUNDARY_GUIDANCE_MAX=0.15
export BOUNDARY_GUIDANCE_START_RATIO=0.35
export BOUNDARY_GUIDANCE_TEMPERATURE=0.05
export BOUNDARY_BRANCH_SCALE=0.10

export ENABLE_BOUNDARY_MODULATION=1
export BOUNDARY_MOD_SCALE=0.10
export BOUNDARY_MOD_START_RATIO=0.75

export ENABLE_TOLERANCE_BAND_LOSS=1
export LAMBDA_BAND=0.02
export BOUNDARY_BAND_T_GATE=200

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

LOG_FILE="${DIFF_STAGE2_LOG_DIR}/train_stage2_bef_sbg_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/08_train_diffusion_stage2_${RATIO_TAG}.pid"

nohup python -u "${BGDIFF_ROOT}/tutorial_train_bef.py" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 08 started: train diffusion stage2"
echo "     ratio:  ${RATIO_TAG}"
echo "     gpu:    ${CUDA_VISIBLE_DEVICES}"
echo "     resume: ${RESUME_PATH}"
echo "     prompt: ${PROMPT_JSON}"
echo "     pid:    $(cat ${PID_FILE})"
echo "     log:    ${LOG_FILE}"
echo "     kill:   kill \$(cat ${PID_FILE})"
