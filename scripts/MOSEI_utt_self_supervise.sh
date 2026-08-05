#!/bin/bash
set -e

# 正确接收运行参数
modality=$1   # 第1个参数: 模态组合 (例如 AVL)
run_idx=$2    # 第2个参数: 运行序号 (例如 1)
gpu=$3        # 第3个参数: GPU 编号 (例如 0)

# 清理旧的 MOSEI 预训练权重，防止因上次中断导致权重污染
rm -rf ./checkpoints/MOSEI_utt_self_supervise_${modality}_run${run_idx}

# 开启 10 折交叉验证循环 (MOSEI 数据量大，跑完需要一定时间)
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=multimodal --model=utt_self_supervise \
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu --image_dir=./shared_image \
--A_type=acoustic --input_dim_a=74 --norm_method=trn --embd_size_a=128 --embd_method_a=maxpool \
--V_type=visual --input_dim_v=35 --embd_size_v=128  --embd_method_v=maxpool \
--L_type=bert_large --input_dim_l=768 --embd_size_l=128 \
--num_thread=8 --corpus=MOSEI --corpus_name=MOSEI \
--output_dim=1 --cls_layers=128,128 --dropout_rate=0.3 \
--niter=20 --niter_decay=20 --verbose --print_freq=50 \
--batch_size=128 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5 \
--name=MOSEI_utt_self_supervise_${modality}_run${run_idx} \
--modality=$modality \
--has_test \
--cvNo=$i --num_classes=1 --random_seed=336"

echo -e "\n====================================================================================="
echo "🚀 正在执行 MOSEI 预训练 | 折数 (Fold): $i/10 | 模态: $modality | GPU: $gpu"
echo "命令详情: $cmd"
echo -e "=====================================================================================\n"

eval $cmd

done