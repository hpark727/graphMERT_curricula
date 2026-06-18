#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/data_prep_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/data_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu

# ---- user config --------------------------------------------------------
REPO_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen"
DATA_DIR="/scratch/gpfs/JHA/hp9084/curricula_gen/output"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"

DATASET="$DATA_DIR/curriculum_dataset_hop_1_2.json"
# -------------------------------------------------------------------------

mkdir -p "$DATA_DIR" "/scratch/gpfs/JHA/hp9084/curricula_gen/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Input dataset: $DATASET"

cd "$REPO_DIR/data"

echo "--- Step 1: Decontamination ---"
python3 decontamination.py \
    --train_questions_path "$DATASET" \
    --ngram_size 18

echo "--- Step 2: Tokenization ---"
python3 tokenization.py \
    --dataset_train_path "${DATASET%.json}_decontaminated.json"

echo "Job finished at $(date)"
echo "Tokenized dataset saved to: $DATA_DIR/tokenized_curriculum_dataset_hop_1_2_decontaminated/"
