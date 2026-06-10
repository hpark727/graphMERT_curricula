#!/bin/bash

set -e

ENV_NAME="bottom_up_SI"

echo "Creating conda environment: $ENV_NAME"

# Create new conda environment with Python 3.11 (compatible with all packages)
conda create -n $ENV_NAME python=3.11 -y

# Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# Install PyTorch with CUDA support first (base dependency)
echo "Installing PyTorch with CUDA support..."
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install core ML packages
echo "Installing core ML packages..."
pip install transformers \
            datasets==4.0.0 \
            tokenizers \
            accelerate==1.9.0 \
            peft==0.16.0 \
            networkx==3.4.2 \
            python-dotenv \
            huggingface_hub

# Install vLLM for fast inference (0.8+ required for Gemma 3 support)
echo "Installing vLLM..."
pip install "vllm>=0.8.0"


# Install additional useful packages for evaluation
echo "Installing additional evaluation packages..."
pip install scikit-learn==1.7.1 \
            pandas==2.2.2 \
            numpy==1.26.4 \
            tqdm==4.67.1 \
            wandb==0.21.0 \
            jsonlines==4.0.0 \
            rouge_score==0.1.2 \
            sacrebleu==2.5.1 \
            evaluate==0.4.5

# Install development and utility packages
echo "Installing utility packages..."
pip install jupyter \
            ipython==9.1.0 \
            matplotlib \
            seaborn

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "sft --help"