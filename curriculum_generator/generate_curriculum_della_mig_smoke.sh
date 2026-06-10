#!/bin/bash
#SBATCH --job-name=curriculum_smoke
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/smoke_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hp9084@princeton.edu

# ---- user config --------------------------------------------------------
MODEL="google/gemma-4-E4B-it"    # fits in 10GB MIG; swap for 27b-it on 80GB nodes
TENSOR_PARALLEL=1                # MIG = single GPU only
NUM_QUESTIONS=10
MAX_K_HOPS=2
BATCH_SIZE=4
DOMAIN="computer networking"

REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
OUTPUT_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen/smoke_output"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"
# -------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR" "/scratch/gpfs/JHA/hp9084/curricula_gen/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Model: $MODEL | GPUs: $TENSOR_PARALLEL | Questions: $NUM_QUESTIONS"

python "$REPO_DIR/curriculum_generator/generate_curriculum_local.py" \
    --model_name "$MODEL" \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --num_questions $NUM_QUESTIONS \
    --max_k_hops $MAX_K_HOPS \
    --batch_size $BATCH_SIZE \
    --domain "$DOMAIN" \
    --output_dir "$OUTPUT_DIR" \
    --hf_cache_dir "$HF_CACHE"

echo "Smoke test finished at $(date)"
