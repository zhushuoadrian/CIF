#!/bin/bash
set -e
gpu=0
run_idx=1
modality="AVL"

for i in $(seq 1 1 10);
do

cmd="python train_baseline.py \
--dataset_mode=multimodal \
--model=utt_self_supervise \
--log_dir=./logs \
--checkpoints_dir=./checkpoints \
--gpu_ids=$gpu \
--corpus_name=SIMS \
--output_dim=1 \
--modality=$modality \
--A_type=acoustic --input_dim_a=33 --embd_size_a=128 \
--V_type=visual --input_dim_v=709 --embd_size_v=128 \
--L_type=text --input_dim_l=768 --embd_size_l=128 \
--name=SIMS_utt_self_supervise_run_${run_idx} \
--batch_size=64 \
--lr=2e-4 \
--niter=20 \
--niter_decay=20 \
--random_seed=336 \
--cvNo=$i"

echo -e "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo -e "-------------------------------------------------------------------------------------\n"
eval $cmd

done