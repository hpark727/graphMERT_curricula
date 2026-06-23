#!/bin/bash
#SBATCH --job-name=gap_fill_redundant
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/gap_fill_redundant_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/gap_fill_redundant_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --time=6:00:00

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

python -m curriculum_generator.fill_coverage_gaps_redundant \
    --full_dataset  "$DATA_DIR/curriculum_dataset_hop_3.json" \
    --verdicts      "$REPO_DIR/curriculum_generator/data/verdicts_gptoss120b.json" \
    --kg_triples    "$REPO_DIR/networks_kg/validated_triples.csv" \
    --output        "$DATA_DIR/curriculum_dataset_gap_fill_redundant.json" \
    --model_name    google/gemma-3-27b-it \
    --hf_cache      "$HF_CACHE" \
    --tensor_parallel_size 2 \
    --batch_size    64 \
    --num_per_triple 10

echo "Done at $(date)"
echo "Next: run grade_curriculum_tinker.py on curriculum_dataset_gap_fill_redundant.json"
