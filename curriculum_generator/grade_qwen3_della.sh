#!/bin/bash
#SBATCH --job-name=grade_qwen3
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/grade_qwen3_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/grade_qwen3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=                  # fill in your princeton email

# ---- user config --------------------------------------------------------
MODEL="Qwen/Qwen3.6-27B"
GRADER_NAME="qwen3"
TENSOR_PARALLEL=2
BATCH_SIZE=64
MAX_MODEL_LEN=8192

REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
DATASET="/scratch/gpfs/JHA/hp9084/curricula_gen/output/curriculum_dataset_hop_1_2.json"
OUTPUT_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen/output"
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
echo "Model: $MODEL | Grader: $GRADER_NAME | GPUs: $TENSOR_PARALLEL"

python "$REPO_DIR/curriculum_generator/grade_curriculum.py" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL" \
    --grader_name "$GRADER_NAME" \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --batch_size $BATCH_SIZE \
    --max_model_len $MAX_MODEL_LEN \
    --hf_cache_dir "$HF_CACHE"

echo "Job finished at $(date)"
