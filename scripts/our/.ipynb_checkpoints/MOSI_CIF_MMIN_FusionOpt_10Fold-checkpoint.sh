#!/bin/bash
set -e

# ------------------------------------------------------------------
# 脚本名称: MOSI_CIF_MMIN_FusionOpt_10Fold.sh
# 功能描述: 运行 MOSI 数据集的 10折交叉验证 (10-Fold CV)
#          每一折训练 40 Epochs (20 niter + 20 niter_decay)
# 使用方法: bash scripts/our/MOSI_CIF_MMIN_FusionOpt_10Fold.sh [GPU_ID]
# 示例:     bash scripts/our/MOSI_CIF_MMIN_FusionOpt_10Fold.sh 0
# ------------------------------------------------------------------

# 1. 获取命令行参数 (默认使用 GPU 0)
GPU_ID=${1:-0}

echo "======================================================"
echo "准备开始: 10折交叉验证 (Fold 1 - Fold 10)"
echo "使用的 GPU: $GPU_ID"
echo "每折轮数: 40 (20 + 20)"
echo "======================================================"

# 2. 循环运行 Fold 1 到 Fold 10
for i in {10..10}; do
    echo ""
    echo "------------------------------------------------------"
    echo "正在运行 Fold $i / 10"
    echo "------------------------------------------------------"
    
    python train_miss.py \
    --dataset_mode=multimodal_miss \
    --model=CIF_MMIN \
    --log_dir=./logs \
    --checkpoints_dir=./checkpoints \
    --gpu_ids=$GPU_ID \
    --image_dir=./shared_image \
    --A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
    --V_type=visual --input_dim_v=47 --embd_size_v=128 --embd_method_v=maxpool \
    --L_type=bert_large --input_dim_l=768 --embd_size_l=128 \
    --num_thread=8 \
    --corpus=MOSI --corpus_name=MOSI \
    --output_dim=1 --cls_layers=128,64 --dropout_rate=0.5 \
    --verbose --print_freq=10 \
    --batch_size=64 --lr=1e-4 --weight_decay=0.00001 \
    --has_test \
    --pretrained_path='checkpoints/MOSI_utt_self_supervise_AVL_run1' \
    --random_seed=336 \
    --align_dim=128 \
    --mamba_d_state=32 \
    --name=name=CIF_MMIN_MOSI_FusionOpt_Mamba_REAL_V1 \
    --ce_weight=1.0 --mse_weight=1.0 --cycle_weight=1.0 --consistent_weight=1.0 \
    --seq_weight=0.5 \
    --run_idx=1 \
    --cvNo=$i \
    --suffix=run_fold${i} \
    --niter=20 --niter_decay=20
    
    echo ">> Fold $i 完成！"
done

echo "======================================================"
echo "10折交叉验证全部完成。"
echo "======================================================"