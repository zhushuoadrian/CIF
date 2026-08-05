#!/bin/bash
set -e  # 遇到错误立即停止，安全第一
gpu=0

for i in {1..10}; do

cmd="python train_miss.py \
--dataset_mode=multimodal \
--model=CIF_MMIN \
--log_dir=./logs \
--checkpoints_dir=./checkpoints \
--gpu_ids=$gpu \
--corpus_name=SIMS \
--output_dim=1 \
--A_type=acoustic --input_dim_a=33 --embd_size_a=128 \
--V_type=visual --input_dim_v=709 --embd_size_v=128 \
--L_type=text --input_dim_l=768 --embd_size_l=128 \
--AE_layers=256,128,64 \
--n_blocks=5 \
--cls_layers=128,64 \
--dropout_rate=0.5 \
--ce_weight=1.0 \
--mse_weight=1.0 \
--consistent_weight=1.0 \
--name=CIF_MMIN_SIMS_Final \
--suffix=run_${gpu} \
--has_test \
--batch_size=32 \
--lr=1e-4 \
--niter=40 \
--niter_decay=40 \
--pretrained_path='checkpoints/SIMS_utt_self_supervise_run_1' \
--cvNo=$i \
--random_seed=336"

echo -e "\n-------------------------------------------------------------------------------------"
echo "Starting Fold $i / 10"
echo "Execute command: $cmd"
echo -e "-------------------------------------------------------------------------------------\n"
eval $cmd

done