#!/bin/bash
'''
Copyright (c) 2025 The Trustees of Princeton University
Authors: Bhishma Dedhia, Yuval Kansal, Niraj K. Jha

Licensed for academic and research use only.
See LICENSE file for full terms.
'''

# Get node information
node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
head_node=($node_array)
head_node_ip=$(ssh $head_node hostname --ip-address)
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
batch_size=16
export LOCAL_RANK=$SLURM_LOCALID
# Calculate gradient accumulation steps
gpu_count=$(nvidia-smi -L | wc -l)
grad_acc=$(($batch_size/(gpu_count * nnodes)))


echo "Number of nodes: $nnodes"
echo "Number of GPUs per node: $gpu_count"
echo "Head node IP: $head_node_ip"

conda activate bottom_up_SI

run_name="qwq_32b_curriculum_training_data_bs16_lr1e-5_epoch8_wd1e-4_$(date +%Y%m%d_%H%M%S)"
torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    trainer.py \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=${grad_acc} \
    --num_train_epochs=8 \
    --train_dataset_path="/curriculum_training_data/tokenized_curriculum_dataset_hop_3_decontaminated/" \
    --model_name=Qwen/QwQ-32B \
    --warmup_ratio=0.05 \
    --report_to="wandb" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="fsdp_config_qwen.json" \
    --bf16=True \
    --save_strategy="no" \
    --eval_strategy="no" \
    --logging_steps=1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="/checkpoints/${run_name}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --use_lora  
