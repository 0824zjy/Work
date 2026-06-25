#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

INFER_LOG_DIR="${LOG_ROOT}/diffusion/infer_bef_sbg_${RATIO_TAG}"
mkdir -p "${INFER_LOG_DIR}"
mkdir -p "${GEN_OUT_DIR}"

cd "${BGDIFF_ROOT}"

export PROJECT_DIR="${BGDIFF_ROOT}"

export CKPT_PATH="${CKPT_PATH:-${DIFF_STAGE2_LOG_DIR}/checkpoints/last.ckpt}"
export PROMPT_JSON="${BEF_PROMPT_JSON}"
export OUT_DIR="${GEN_OUT_DIR}"

export DEVICE="cuda:0"
export BATCH_SIZE=1
export N_SAMPLES="${N_SAMPLES:-2}"
export NUM_WORKERS=4
export IMG_SIZE=384

export SAMPLE_SEED_BASE="${SAMPLE_SEED_BASE:-0}"

export DDIM_STEPS="${DDIM_STEPS:-70}"
export CFG="${CFG:-9.0}"

export USE_IMAGE_CONTROL=0

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

export ENABLE_TOLERANCE_BAND_LOSS=0
export LAMBDA_BAND=0.0

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

LOG_FILE="${INFER_LOG_DIR}/infer_bef_sbg_${RATIO_TAG}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/09_infer_diffusion_bef_sbg_${RATIO_TAG}.pid"

nohup python -u "${BGDIFF_ROOT}/tutorial_inference_bef.py" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 09 started: infer BEF-SBG diffusion samples"
echo "     ratio:   ${RATIO_TAG}"
echo "     gpu:     ${CUDA_VISIBLE_DEVICES}"
echo "     ckpt:    ${CKPT_PATH}"
echo "     prompt:  ${PROMPT_JSON}"
echo "     out_dir: ${OUT_DIR}"
echo "     pid:     $(cat ${PID_FILE})"
echo "     log:     ${LOG_FILE}"
echo "     kill:    kill \$(cat ${PID_FILE})"
