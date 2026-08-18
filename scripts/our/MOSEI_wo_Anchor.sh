#!/bin/bash
set -e

# 接收运行参数
run_idx=$1  # 运行序号
gpu=$2      # GPU编号

# 强行绑定 GPU，解决多进程初始化冲突
export CUDA_VISIBLE_DEVICES=$gpu

# MOSEI 10折交叉验证循环 (注：若MOSEI采用固定划分，这里循环次数视你的需求调整)
for i in {1..10}; do

# MOSEI 专属配置 (消融版)
# 移除了 --consistent_weight 和 --pretrained_path
cmd="python train_miss.py --dataset_mode=multimodal --model=CIF_MMIN \
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=0 --image_dir=./shared_image \
--A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
--V_type=visual --input_dim_v=35 --embd_size_v=128  --embd_method_v=maxpool \
--L_type=bert_large --input_dim_l=768 --embd_size_l=128 \
--AE_layers=256,128,64 --n_blocks=5 --num_thread=8 --corpus=MOSEI --corpus_name=MOSEI \
--ce_weight=1.0 --mse_weight=8.0 \
--output_dim=1 --cls_layers=128,64 --dropout_rate=0.5 \
--niter=20 --niter_decay=20 --verbose --print_freq=50 \
--batch_size=64 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5 \
--name=CIF_MMIN_MOSEI_wo_Anchor --suffix=block_5_run_${gpu}_${run_idx} --has_test \
--cvNo=$i --num_classes=1 --random_seed=336"

echo -e "\n-------------------------------------------------------------------------------------"
echo "🚀 正在执行 MOSEI 消融实验 (w/o Stage-IAnchor) | 折数 (Fold): $i/10 | GPU: $gpu"
echo "Execute command: $cmd"
echo -e "-------------------------------------------------------------------------------------\n"

eval $cmd

done