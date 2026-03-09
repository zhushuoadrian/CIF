#!/bin/bash
set -e

# 正确接收基线的 3 个参数
modality=$1   # 第1个参数: AVL
run_idx=$2    # 第2个参数: 1
gpu=$3        # 第3个参数: 0

# 清理旧的、错误的预训练权重
rm -rf ./checkpoints/MOSI_utt_self_supervise_${modality}_run${run_idx}

for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal --model=utt_self_supervise \
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu --image_dir=./shared_image \
--A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
--V_type=visual --input_dim_v=47 --embd_size_v=128  --embd_method_v=maxpool \
--L_type=bert_large --input_dim_l=768 --embd_size_l=128 \
--num_thread=8 --corpus=MOSI --corpus_name=MOSI \
--output_dim=1 --cls_layers=128,128 --dropout_rate=0.3 \
--niter=20 --niter_decay=20 --verbose --print_freq=10 \
--batch_size=64 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5 \
--name=MOSI_utt_self_supervise_${modality}_run${run_idx} \
--modality=$modality \
--has_test \
--cvNo=$i --num_classes=1 --random_seed=336"

echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo -e "-------------------------------------------------------------------------------------\n"
eval $cmd

done