#!/bin/bash
set -x
export PYTHON_UNBUFFERED=1
export PYTHONPATH=../dolomite-engine
MAX_LENGTH=4096
MODEL_PATH=stickbreaking_1b_norem
python -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--tasks pile_10k \
	2>&1

exit
python -m lm_eval \
	--model hf \
    --model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=exported_huggingface_VT,use_flash_attention_2=True"  \
	--batch_size 1 \
	--tasks pile_10k \
	2>&1


