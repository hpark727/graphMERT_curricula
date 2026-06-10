#!/bin/bash
#SBATCH --job-name=curriculum_gen
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/curriculum_gen_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/curriculum_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8             # 80GB nodes have 12 CPU cores per GPU; 8 is safe for 2 GPUs
#SBATCH --gres=gpu:2                  # Gemma 3 27B needs ~27GB/GPU in bf16 + KV cache → 80GB nodes
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=                  # fill in your princeton email

# ---- user config --------------------------------------------------------
MODEL="google/gemma-3-27b-it"                # any HF model ID
TENSOR_PARALLEL=2                            # must match --gres=gpu:N above
NUM_QUESTIONS=24000
MAX_K_HOPS=3
BATCH_SIZE=16
DOMAIN="computer networking"

REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
OUTPUT_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen/output"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"
# -------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR" "/scratch/gpfs/JHA/hp9084/curricula_gen/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

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

echo "Job finished at $(date)"
