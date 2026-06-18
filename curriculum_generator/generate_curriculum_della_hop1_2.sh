#!/bin/bash
#SBATCH --job-name=curriculum_gen_hop1_2
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/curriculum_gen_hop1_2_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/curriculum_gen_hop1_2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=                  # fill in your princeton email

# ---- user config --------------------------------------------------------
MODEL="google/gemma-3-27b-it"
TENSOR_PARALLEL=2
NUM_QUESTIONS=20000                          # 12k already exist; will resume to 20k total
MAX_K_HOPS=2                                # only 1-hop and 2-hop questions
BATCH_SIZE=64
DOMAIN="computer networking"

REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
OUTPUT_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen/output"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"

# Existing 1-2 hop dataset to resume from (already has ~12k questions)
OUTPUT_FILE="$OUTPUT_DIR/curriculum_dataset_hop_1_2.json"
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
echo "Model: $MODEL | GPUs: $TENSOR_PARALLEL | Target: $NUM_QUESTIONS | Max hops: $MAX_K_HOPS"

python "$REPO_DIR/curriculum_generator/generate_curriculum_local.py" \
    --model_name "$MODEL" \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --num_questions $NUM_QUESTIONS \
    --max_k_hops $MAX_K_HOPS \
    --batch_size $BATCH_SIZE \
    --domain "$DOMAIN" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --hf_cache_dir "$HF_CACHE"

echo "Job finished at $(date)"
