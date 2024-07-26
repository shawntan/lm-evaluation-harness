#!/bin/bash

export PYTHON_UNBUFFERED=1

../dolomite-engine/scripts/export.sh export.yml
lm_eval \
	--model dolomite \
    	--tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,lambada_openai,boolq,copa,hellaswag,winogrande,race,wikitext \
	--model_args "dtype=bfloat16,max_length=4096,pretrained=exported_huggingface_model,parallelize=True"  \
	--batch_size 1 \
	--device cuda --num_fewshot 0 2>&1

	# --model_args "dtype=bfloat16,max_length=4096,pretrained=exported_huggingface_model,parallelize=True,use_flash_attention_2=True,use_padding_free_transformer=True"  \
    	# --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,lambada_openai,boolq,copa,hellaswag,winogrande,race,wikitext \
	# --model openllama \
# pretrained=../scattermoe/examples/converted/,\
    # --tasks wikitext \
    # --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,lambada_standard,boolq,copa,hellaswag,winogrande,race \
   #  --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande,triviaqa,webqs,mathqa,wikitext,lambada_openai,race \
    # --model hf-causal-experimental \
    # --model openllama_sb \

# lm_eval \
#     --model hf \
#     --model_args \
# pretrained=mistralai/Mixtral-8x7B-v0.1,\
# dtype=bfloat16,max_length=2048,\
# parallelize=True,\
#     --batch_size 32 \
#     --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext \
#     --device cuda --num_fewshot 0  2>&1

