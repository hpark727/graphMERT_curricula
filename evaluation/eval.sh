#! /bin/bash
conda activate bottom_up_SI

cd evaluation/lm-evaluation-harness
pip install -e .
#parallel scaling
lm_eval --model vllm --model_args pretrained=bottom-up-superintelligence/qwq_med_3,dtype=bfloat16,tensor_parallel_size=8 --tasks icdbench --batch_size auto --apply_chat_template --output_path /eval_outputs/qwq_med_3/parallel/ --log_samples --gen_kwargs "max_gen_toks=32768,temperature=0.6" 
#sequential scaling
lm_eval --model vllm --model_args pretrained=bottom-up-superintelligence/qwq_med_3,dtype=bfloat16,tensor_parallel_size=8 --tasks icdbench --batch_size auto --apply_chat_template --output_path /eval_outputs/qwq_med_3/sequential/ --log_samples "max_gen_toks=32768,max_tokens_thinking=auto,thinking_n_ignore=5,thinking_n_ignore_str=hmm,temperature=0.0"