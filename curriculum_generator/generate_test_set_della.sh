#!/bin/bash
#SBATCH --job-name=test_set_gen
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/test_set_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/test_set_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=12:00:00

REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
DATA_DIR="$REPO_DIR/output"
HF_CACHE="$REPO_DIR/.cache/huggingface"

mkdir -p "$DATA_DIR" "$REPO_DIR/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

cd "$REPO_DIR"

# Generate 3000 per hop (3/4/5) → expect ~2300 each after GPT grading at 76%
python -m curriculum_generator.generate_test_set \
    --output        "$DATA_DIR/test_set_raw.json" \
    --model_name    google/gemma-3-27b-it \
    --hf_cache      "$HF_CACHE" \
    --tensor_parallel_size 2 \
    --batch_size    64 \
    --num_per_hop   3000 \
    --hop_counts    3 4 5

echo "Done at $(date)"
echo "Next: run grade_curriculum_tinker.py on test_set_raw.json"
