#!/bin/bash
#SBATCH --job-name=sft_qwen3_14b
#SBATCH --output=/scratch/gpfs/JHA/hp9084/sft_training/logs/sft_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/sft_training/logs/sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=                  # fill in your princeton email

# ---- user config --------------------------------------------------------
REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
DATASET_PATH="/scratch/gpfs/JHA/hp9084/curricula_gen/output/tokenized_curriculum_dataset_final"
CHECKPOINT_DIR="/scratch/gpfs/JHA/hp9084/sft_training/checkpoints"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"
# -------------------------------------------------------------------------

mkdir -p "$CHECKPOINT_DIR" "/scratch/gpfs/JHA/hp9084/sft_training/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600
export LOCAL_RANK=$SLURM_LOCALID

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi -L | wc -l) | Dataset: $DATASET_PATH"

node_array=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nnodes=$(echo $node_array | wc -w)
head_node=($node_array)
head_node_ip=$(ssh $head_node hostname --ip-address)
gpu_count=$(nvidia-smi -L | wc -l)

batch_size=16
grad_acc=$(($batch_size / ($gpu_count * $nnodes)))

run_name="qwen3_14b_sft_final_bs${batch_size}_lr1e-5_epoch8_$(date +%Y%m%d_%H%M%S)"

echo "Nodes: $nnodes | GPUs/node: $gpu_count | Grad acc: $grad_acc | Run: $run_name"

torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpu_count \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    "$REPO_DIR/training/trainer.py" \
    --model_name=Qwen/Qwen3-14B \
    --block_size=32768 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=${grad_acc} \
    --num_train_epochs=8 \
    --train_dataset_path="$DATASET_PATH" \
    --warmup_ratio=0.05 \
    --report_to="wandb" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="$REPO_DIR/training/fsdp_config_qwen.json" \
    --bf16=True \
    --save_strategy="no" \
    --eval_strategy="no" \
    --logging_steps=1 \
    --lr_scheduler_type="cosine" \
    --learning_rate=1e-5 \
    --weight_decay=1e-4 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="$CHECKPOINT_DIR/$run_name" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --use_lora

echo "Job finished at $(date)"
