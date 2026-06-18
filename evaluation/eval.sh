#! /bin/bash
# ---- user config --------------------------------------------------------
MODEL_PATH="/scratch/gpfs/JHA/hp9084/sft_training/checkpoints/<run_name>"  # fill in after training
EVAL_OUTPUT="/scratch/gpfs/JHA/hp9084/sft_training/eval_outputs"
# -------------------------------------------------------------------------

module purge
module load anaconda3/2024.2
conda activate bottom_up_SI

cd evaluation/lm-evaluation-harness
pip install -e .

mkdir -p "$EVAL_OUTPUT"

# greedy decoding
lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=4 \
    --tasks icdbench \
    --batch_size auto \
    --apply_chat_template \
    --output_path "$EVAL_OUTPUT/greedy/" \
    --log_samples \
    --gen_kwargs "max_gen_toks=8192,temperature=0.0"

# sampling (majority vote / pass@k)
lm_eval \
    --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=bfloat16,tensor_parallel_size=4 \
    --tasks icdbench \
    --batch_size auto \
    --apply_chat_template \
    --output_path "$EVAL_OUTPUT/sampling/" \
    --log_samples \
    --gen_kwargs "max_gen_toks=8192,temperature=0.6"