#!/bin/bash
set -e
run_idx=$1
gpu=$2

# 强行绑定当前任务到你指定的显卡上，防止进程级别的 CUDA 初始化冲突
export CUDA_VISIBLE_DEVICES=$gpu

for i in {1..10}; do
# 注意：这是 w/o Stage-IAnchor 的消融实验脚本

# 注意：此处的 --gpu_ids 改为 0，以匹配上面 export 后的单卡逻辑视野
cmd="python train_miss.py --dataset_mode=multimodal --model=CIF_MMIN \
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=0 --image_dir=./shared_image \
--A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
--V_type=visual --input_dim_v=47 --embd_size_v=128  --embd_method_v=maxpool \
--L_type=bert_large --input_dim_l=768 --embd_size_l=128 \
--AE_layers=256,128,64 --n_blocks=5 --num_thread=8 --corpus=MOSI --corpus_name=MOSI \
--ce_weight=1.0 --mse_weight=8.0 \
--output_dim=1 --cls_layers=128,64 --dropout_rate=0.5 \
--niter=20 --niter_decay=20 --verbose --print_freq=10 \
--batch_size=64 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5 \
--name=CIF_MMIN_MOSI_wo_Anchor --suffix=block_5_run_${gpu}_${run_idx} --has_test \
--cvNo=$i --num_classes=1 --random_seed=336"

echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo -e "-------------------------------------------------------------------------------------\n"

eval $cmd

done