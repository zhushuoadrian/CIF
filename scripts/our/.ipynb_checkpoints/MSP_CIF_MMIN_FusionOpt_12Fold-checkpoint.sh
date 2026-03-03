#!/bin/bash

# 获取 GPU ID
gpu_ids=${1:-0}
CORPUS="MSP"

# =========================================================
# 1. 特征文件名 (根据您之前 ls 确认的结果)
# =========================================================
A_TYPE="comparE"       # 对应 A/comparE.h5
L_TYPE="bert_large"    # 对应 L/bert_large.h5
V_TYPE="denseface"     # 对应 V/denseface.h5

# =========================================================
# 2. 核心参数修改 (根据论文要求)
# =========================================================
# [关键] 学习率改为 0.0002
LR=0.0002

# [关键] Epoch设置: 总共 40 轮
# niter: 保持初始学习率的轮数 (20)
# niter_decay: 线性衰减到0的轮数 (20)
# Total = 20 + 20 = 40
NITER=20
NITER_DECAY=20

# 12折交叉验证
TOTAL_CV=12 

# =========================================================
# MSP-IMPROV 维度设置
# =========================================================
INPUT_DIM_A=130
INPUT_DIM_L=1024
INPUT_DIM_V=342

OUTPUT_DIM=4      # 4分类
CE_WEIGHT=1.0
MSE_WEIGHT=0.0

BATCH_SIZE=32     # 论文没提，32是比较稳妥的选择
NAME="CIF_MMIN_MSP_FusionOpt_SoftGate"

# =========================================================
# 循环 1 到 12 折
# =========================================================
for cv in $(seq 1 $TOTAL_CV)
do
    echo "----------------------------------------------------------------"
    echo "Start Training Fold $cv / $TOTAL_CV for $CORPUS ..."
    echo "----------------------------------------------------------------"

    python train_miss.py \
        --name ${NAME} \
        --model CIF_MMIN \
        --dataset_mode multimodal \
        --corpus_name ${CORPUS} \
        --gpu_ids ${gpu_ids} \
        --cvNo ${cv} \
        \
        --niter ${NITER} \
        --niter_decay ${NITER_DECAY} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        \
        --has_test \
        \
        --input_dim_a ${INPUT_DIM_A} \
        --input_dim_l ${INPUT_DIM_L} \
        --input_dim_v ${INPUT_DIM_V} \
        --output_dim ${OUTPUT_DIM} \
        \
        --A_type ${A_TYPE} \
        --L_type ${L_TYPE} \
        --V_type ${V_TYPE} \
        \
        --embd_size_a 128 \
        --embd_size_v 128 \
        --embd_size_l 128 \
        \
        --cycle_weight 0.0 \
        --consistent_weight 0.0 \
        --ce_weight ${CE_WEIGHT} \
        --mse_weight ${MSE_WEIGHT} \
        \
        --cls_layers "128,128" \
        --dropout_rate 0.3 \
        \
        --mamba_d_state 16 \
        --test_noise_level 0.0 \
        \
        --pretrained_path ./checkpoints/MSP_utt_self_supervise_AVL_run1 \
        --checkpoints_dir ./checkpoints \
        --print_freq 10 \
        --verbose
done

echo "========================================================"
echo "All 12 folds for MSP-IMPROV finished!"
echo "========================================================"