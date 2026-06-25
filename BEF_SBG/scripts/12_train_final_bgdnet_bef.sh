#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

mkdir -p "${LOG_ROOT}/segmentation"
mkdir -p "${FINAL_SEG_SAVE_DIR}"

# Final BGDNet training also disables cuDNN in this environment.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"

# batchsize=4 默认不冻结 BN。
# 如果你后续改 FINAL_BATCHSIZE=1，则建议同时设置 BGDNET_FREEZE_BN=1，
# 但 train_BGDNet_BEF.py 也需要实现 freeze BN helper 才会生效。
export BGDNET_FREEZE_BN="${BGDNET_FREEZE_BN:-0}"

export FINAL_BATCHSIZE="${FINAL_BATCHSIZE:-4}"

export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
export TORCH_DISABLE_ADDR2LINE=1

LOG_FILE="${LOG_ROOT}/segmentation/train_final_bgdnet_bef_${RATIO_TAG}_bs${FINAL_BATCHSIZE}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/12_train_final_bgdnet_bef_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/segmentation/train_BGDNet_BEF.py" \
  --weighted_train_json "${WEIGHTED_TRAIN_JSON}" \
  --test_path "${ISIC_TEST_ROOT}" \
  --epoch 200 \
  --lr 1e-4 \
  --batchsize "${FINAL_BATCHSIZE}" \
  --img_size 352 \
  --train_save "${FINAL_SEG_SAVE_DIR}" \
  --alpha 1.0 \
  --beta 0.4 \
  --clip 0.5 \
  --augmentation False \
  --num_workers 8 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 12 started: train final BGDNet-BEF"
echo "     ratio:      ${RATIO_TAG}"
echo "     gpu:        ${CUDA_VISIBLE_DEVICES}"
echo "     batchsize:  ${FINAL_BATCHSIZE}"
echo "     cudnn off:  ${BGDNET_DISABLE_CUDNN}"
echo "     train_json: ${WEIGHTED_TRAIN_JSON}"
echo "     save_dir:   ${FINAL_SEG_SAVE_DIR}"
echo "     pid:        $(cat ${PID_FILE})"
echo "     log:        ${LOG_FILE}"
echo "     kill:       kill \$(cat ${PID_FILE})"
