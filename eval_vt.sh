#!/bin/bash
set -x
export PYTHON_UNBUFFERED=1
export PYTHONPATH=~/dolomite-engine
# ln -s /proj/checkpoints/mayank/1b-exp15-reset-p2-decay/global_step62500/model{.pt,}
# 
# ../experimental/dolomite-engine/scripts/unshard.sh export_vanilla.yml
MAX_LENGTH=$((4096 * 1))
# MODEL_PATH="TinyLlama/TinyLlama_v1.1"
MODEL_PATH=granite_3b
accelerate launch -m lm_eval \
    --model hf \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True"  \
    --batch_size 16 \
	--tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,race,sciq,winogrande,wikitext\
	2>&1
exit


# TODO check if padding changes anything
accelerate launch -m lm_eval \
    --model hf \
	--tasks mmlu --num_fewshot 0 \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True"  \
    --batch_size 32 \
    --device cuda  2>&1
accelerate launch -m lm_eval \
    --model hf \
	--tasks mmlu --num_fewshot 5 \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True"  \
    --batch_size 8 \
    --device cuda 2>&1


    # --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=exported_huggingface_VT,use_flash_attention_2=True"  \
	# --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext,mmlu \
