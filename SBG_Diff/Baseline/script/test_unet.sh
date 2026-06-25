#!/bin/bash

cd /data/zjy_work/Baseline || exit 1

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

mkdir -p /data/zjy_work/Baseline/log
mkdir -p /data/zjy_work/Baseline/model_out/ISIC2018_eval_seed1/UNet_real_gen_softonly_real5_syn15_seed1

LOG_FILE="/data/zjy_work/Baseline/log/testUNet_real_gen_softonly_real5_syn15_seed1_$(date +%Y%m%d_%H%M%S).log"

nohup python -u test_seg_baseline.py \
  --model unet \
  --testsize 352 \
  --pth_path /data/zjy_work/Baseline/model_out/ISIC2018_ours_Seed1/UNet_real_gen_softonly_real5_syn15_seed1/unet-best.pth \
  --test_data_path /data/zjy_work/ISIC2018/test/ \
  --test_list /data/zjy_work/data_txt/test.txt \
  --out_dir /data/zjy_work/Baseline/model_out/ISIC2018_eval_seed1/UNet_real_gen_softonly_real5_syn15_seed1/ \
  --device cuda:0 \
  > "${LOG_FILE}" 2>&1 &

echo "UNet testing started."
echo "PID: $!"
echo "Log: ${LOG_FILE}"
