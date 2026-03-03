#!/bin/bash
set -e

# ------------------------------------------------------------------
# 脚本名称: MOSI_CIF_MMIN_FusionOpt.sh
# 功能描述: 运行基于 Mamba 增强和动态 Query 融合 (FusionOpt) 的 CIF-MMIN 实验
# 使用方法: bash scripts/our/MOSI_CIF_MMIN_FusionOpt.sh [运行次数] [GPU_ID]
# 示例:     bash scripts/our/MOSI_CIF_MMIN_FusionOpt.sh 5 0
# ------------------------------------------------------------------

# 1. 获取命令行参数
# ${1:-1} 表示: 默认跑 1 次
# ${2:-0} 表示: 默认用 GPU 0
NUM_RUNS=${1:-1}
GPU_ID=${2:-0}

echo "======================================================"
echo "准备开始: 在 GPU $GPU_ID 上运行 $NUM_RUNS 次 FusionOpt 实验"
echo "======================================================"

# 2. 循环运行实验
for r in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "------------------------------------------------------"
    echo "正在运行第 $r / $NUM_RUNS 次实验 (Run Index: $r)"
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
    --batch_size=64 --lr=2e-4 --weight_decay=1e-5 \
    --has_test \
    --pretrained_path='checkpoints/MOSI_utt_self_supervise_AVL_run1' \
    --random_seed=336 \
    --align_dim=128 \
    --mamba_d_state=16 \
    --name=CIF_MMIN_MOSI_FusionOpt \
    --ce_weight=1.0 --mse_weight=1.0 --cycle_weight=1.0 --consistent_weight=1.0 \
    --seq_weight=0.5 \
    --run_idx=$r \
    --cvNo=1 \
    --suffix=run_fold1_idx${r} \
    --niter=20 --niter_decay=20
    
    echo ">> 第 $r 次 FusionOpt 实验完成！"
done

echo "======================================================"
echo "所有 $NUM_RUNS 次实验已全部完成。"
echo "======================================================"
