#!/usr/bin/env bash
set -e

source /data/zjy_work/Work3_BEF_SBG/scripts/00_common.sh

mkdir -p "${LOG_ROOT}/segmentation"
mkdir -p "${FINAL_EVAL_DIR}"

# BGDNet testing also disables cuDNN in this environment.
export BGDNET_DISABLE_CUDNN="${BGDNET_DISABLE_CUDNN:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

PTH_PATH="${PTH_PATH:-${FINAL_SEG_SAVE_DIR}/BGDNet-BEF-best.pth}"

LOG_FILE="${LOG_ROOT}/segmentation/test_final_bgdnet_bef_${RATIO_TAG}_cudnn${BGDNET_DISABLE_CUDNN}_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="${PID_DIR}/13_test_final_bgdnet_bef_${RATIO_TAG}.pid"

nohup python -u "${WORK_ROOT}/segmentation/test_BGDNet_BEF.py" \
  --pth_path "${PTH_PATH}" \
  --test_data_path "${ISIC_TEST_ROOT}" \
  --out_dir "${FINAL_EVAL_DIR}" \
  --testsize 352 \
  --device cuda:0 \
  --boundary_tolerance 2 \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

echo "[OK] Step 13 started: test final BGDNet-BEF"
echo "     ratio:     ${RATIO_TAG}"
echo "     gpu:       ${CUDA_VISIBLE_DEVICES}"
echo "     cudnn off: ${BGDNET_DISABLE_CUDNN}"
echo "     pth:       ${PTH_PATH}"
echo "     out_dir:   ${FINAL_EVAL_DIR}"
echo "     pid:       $(cat ${PID_FILE})"
echo "     log:       ${LOG_FILE}"
echo "     kill:      kill \$(cat ${PID_FILE})"
