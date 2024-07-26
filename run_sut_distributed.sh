#!/bin/bash

export PYTHON_UNBUFFERED=1

MODEL_PATH=../out/jetsut_se_aux0.001_256E8K512H/


MODEL_ARGS="dtype=bfloat16,max_length=2048,pretrained=$MODEL_PATH,parallelize=True"
# MODEL_PARAMS="$LARGE_PARAMS${MODEL_ARGS},class_name=llama_sb_big"
MODEL_PARAMS="${MODEL_ARGS},class_name=jetsut_se,\
moe_num_experts=256,moe_top_k=8,\
hidden_size=1024,num_attention_heads=32,ffn_hidden_size=512,num_hidden_layers=24"
export PYTHON_UNBUFFERED=1
lm_eval \
	--model sut \
	--model_args $MODEL_PARAMS \
    --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,wikitext,boolq,copa,hellaswag,winogrande,race \
	--batch_size 32 \
	--device cuda --num_fewshot 0 2>&1

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

