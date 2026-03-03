#!/bin/bash

# =================================================================
# 脚本名称: IEMOCAP_CIF_MMIN_PaperSettings.sh
# 功能: 严格复现论文参数 (LR=0.0002, 40 Epochs) + 修复变量缺失问题
# =================================================================

# 1. 显卡 ID
GPU_ID=0

# 2. 实验名称 (必须定义！否则报错)
EXP_NAME="CIF_IEMOCAP_Mamba_Reproduction"

# 3. 预训练权重路径 (根据您之前的日志确认的路径)
PRETRAINED_PATH="./checkpoints/CAP_utt_self_supervise_AVL_run1"

# 4. 数据维度 (IEMOCAP)
INPUT_DIM_A=130
INPUT_DIM_V=342
INPUT_DIM_L=1024

# 5. 模型结构参数
EMBED_DIM=128
CLS_LAYERS="128,128"

# 6. [论文参数核心区] 
# ---------------------------------------------------------------
# 学习率: 0.0002 (比默认的小5倍，更稳)
LR=0.0002
# Epoch: 前20个保持，后20个衰减 (总共40)
NITER=20
NITER_DECAY=20
# ---------------------------------------------------------------

# 7. 辅助 Loss 权重 (微调)
MSE_WEIGHT=0.1
CYCLE_WEIGHT=0.1
CONSISTENT_WEIGHT=0.0

echo "🚀 开始运行 IEMOCAP 论文复现实验"
echo "   实验名称: $EXP_NAME"
echo "   学习率: $LR"
echo "   Epochs: $((NITER+NITER_DECAY))"

# 8. 循环 10 折
for cv in {8..10}
do
    echo "------------------------------------------------------"
    echo ">>> Processing Fold $cv / 10 ..."
    echo "------------------------------------------------------"

    python train_miss.py \
        --name ${EXP_NAME} \
        --dataset_mode multimodal_miss \
        --model CIF_MMIN \
        --gpu_ids ${GPU_ID} \
        --corpus_name IEMOCAP \
        --output_dim 4 \
        --cvNo ${cv} \
        \
        --pretrained_path ${PRETRAINED_PATH} \
        \
        --A_type comparE \
        --V_type denseface \
        --L_type bert_large \
        \
        --batch_size 32 \
        \
        --lr ${LR} \
        --niter ${NITER} \
        --niter_decay ${NITER_DECAY} \
        --dropout_rate 0.3 \
        --weight_decay 1e-4 \
        \
        --input_dim_a ${INPUT_DIM_A} \
        --input_dim_v ${INPUT_DIM_V} \
        --input_dim_l ${INPUT_DIM_L} \
        --embd_size_a ${EMBED_DIM} \
        --embd_size_v ${EMBED_DIM} \
        --embd_size_l ${EMBED_DIM} \
        \
        --AE_layers "128,64,32" \
        --cls_layers ${CLS_LAYERS} \
        \
        --mse_weight ${MSE_WEIGHT} \
        --cycle_weight ${CYCLE_WEIGHT} \
        --consistent_weight ${CONSISTENT_WEIGHT} \
        --ce_weight 1.0 \
        \
        --mamba_d_state 16 \
        --align_dim 128 \
        --has_test \
        --verbose
done