#!/bin/bash
set -x
export PYTHON_UNBUFFERED=1
export PYTHONPATH=../dolomite-engine
# ../dolomite-engine/scripts/unshard.sh export.yml
# exit
MAX_LENGTH=4096

MODEL_PATH="google/gemma-2-2b"
	# --tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,race,sciq,winogrande,wikitext\
# accelerate launch -m lm_eval \
# 	--model hf \
# 	--model_args "dtype=bfloat16,pretrained=$MODEL_PATH,use_flash_attention_2=True" \
# 	--batch_size 16 \
# 	--tasks arc_easy,arc_challenge,hellaswag,openbookqa,piqa,race,sciq,winogrande,wikitext\
# 	2>&1
# accelerate launch -m lm_eval \
# 	--model hf \
# 	--model_args "dtype=bfloat16,pretrained=$MODEL_PATH,use_flash_attention_2=True" \
# 	--batch_size 16 \
# 	--tasks mmlu --num_fewshot 0 \
# 	2>&1 

accelerate launch -m lm_eval \
	--model hf \
	--model_args "dtype=bfloat16,pretrained=$MODEL_PATH,use_flash_attention_2=True" \
	--batch_size 4 \
	--tasks mmlu --num_fewshot 5 \
	2>&1 

exit

python -m lm_eval \
	--model hf \
	--model_args "dtype=bfloat16,pretrained=$MODEL_PATH,use_flash_attention_2=True" \
	--batch_size 16 \
	--tasks mmlu --num_fewshot 5 \
	2>&1  > smollm_mmlu
exit
accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--tasks wikitext \
	2>&1

exit

accelerate launch -m lm_eval \
    --model dolomite \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True"  \
    --batch_size 1 \
	--tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext,lambada_openai\
	2>&1


accelerate launch -m lm_eval \
    --model dolomite \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True"  \
    --batch_size 1 \
    --device cuda \
	--tasks mmlu --num_fewshot 5 \
	2>&1


    # --tasks hendrycks_math --num_fewshot 5 \
	# --model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True"  \
# accelerate launch -m lm_eval \
#     --model dolomite \
# 	--model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_padding_free_transformer=False"  \
#     --batch_size 1 \
#     --device cuda \
#     --tasks gsm8k --output_path out/ --log_samples

