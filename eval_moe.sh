#!/bin/bash
set -x
export PYTHON_UNBUFFERED=1
export PYTHONPATH=/u/shawntan/proj/experimental/dolomite-engine
/u/shawntan/proj/experimental/dolomite-engine/scripts/unshard.sh export.yml
MAX_LENGTH=4096

# MODEL_PATH=/proj/checkpoints/shawntan/stickbreaking-1b-newkernel/hf
MODEL_PATH=/proj/checkpoints/shawn/varmoe-1b-p1/hf

accelerate launch -m lm_eval \
	--model hf \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH" \
	--batch_size 1 \
	--tasks wikitext
exit
accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 2)),pretrained=$MODEL_PATH" \
	--batch_size 1 \
	--tasks wikitext


accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--tasks wikitext

exit
accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--task mmlu --num_fewshot 5\

accelerate launch -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--task mmlu --num_fewshot 0\



exit
accelerate launch -m lm_eval \
    --model dolomite \
    --model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True"  \
    --batch_size 1 \
	--tasks arc_easy,arc_challenge,boolq,copa,hellaswag,openbookqa,piqa,race,sciq,winogrande,wikitext\
	2>&1
exit
# accelerate launch -m lm_eval \
# 	--model dolomite \
# 	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
# 	--batch_size 1 \
# 	--task mmlu --num_fewshot 5\
# 	2>&1
exit
exit

#	--tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext,lambada_openai\

python -m lm_eval \
	--model dolomite \
	--model_args "dtype=bfloat16,max_length=$((MAX_LENGTH)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True" \
	--batch_size 1 \
	--task \
	2>&1
exit


    # --tasks hendrycks_math --num_fewshot 5 \
	# --model_args "dtype=bfloat16,max_length=$((MAX_LENGTH * 4)),pretrained=$MODEL_PATH,use_flash_attention_2=True,use_padding_free_transformer=True"  \
# accelerate launch -m lm_eval \
#     --model dolomite \
# 	--model_args "dtype=bfloat16,max_length=$MAX_LENGTH,pretrained=$MODEL_PATH,use_padding_free_transformer=False"  \
#     --batch_size 1 \
#     --device cuda \
#     --tasks gsm8k --output_path out/ --log_samples

