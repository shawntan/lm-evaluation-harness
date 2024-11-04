#!/bin/bash
set -x
export PYTHON_UNBUFFERED=1
export PYTHONPATH=../dolomite-engine
# ../dolomite-engine/scripts/unshard.sh export.yml
# exit
MAX_LENGTH=4096
MODEL_PATH=/proj/checkpoints/shawntan/stickbreaking-3b-256gpu-p2-decay25/hf
# MODEL_PATH=exported_huggingface_VT
#--tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,race,sciq,winogrande,wikitext\

accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True" \
	--tasks gsm8k_cot \
	--batch_size 1 --output_path out/sb_3b_decay25_gsm8k_cot
exit
accelerate launch -m lm_eval \
    --model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True" \
    --tasks minerva_math \
	--log_samples --batch_size 1 --output_path out/
accelerate launch -m lm_eval \
    --model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True" \
	--tasks hendrycks_math --num_fewshot 5 \
	--log_samples --batch_size 1 --output_path out/

mkdir -p out/sb_1b_gsm8k


accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True" \
	--tasks gsm8k --num_fewshot 5 \
	--log_samples \
	--batch_size 1 --output_path out/sb_1b_gsm8k

