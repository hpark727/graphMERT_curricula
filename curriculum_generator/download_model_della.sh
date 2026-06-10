#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --output=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/hf_download_%j.out
#SBATCH --error=/scratch/gpfs/JHA/hp9084/curricula_gen/logs/hf_download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=cpu

# ---- user config --------------------------------------------------------
MODEL="google/gemma-3-27b-it"
HF_CACHE="/scratch/gpfs/JHA/hp9084/curricula_gen/.cache/huggingface"
HF_TOKEN=""   # paste your HuggingFace token (required for gated models)
# -------------------------------------------------------------------------

mkdir -p "$HF_CACHE" "/scratch/gpfs/JHA/hp9084/curricula_gen/logs"

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

export HF_HOME="$HF_CACHE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "Downloading $MODEL to $HF_CACHE ..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', cache_dir='$HF_CACHE', token='$HF_TOKEN')
"
echo "Download complete at $(date)"
