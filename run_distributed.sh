#!/bin/bash

export PYTHON_UNBUFFERED=1
HEADS=32
LARGE_PARAMS="num_attention_heads=$HEADS,\
num_key_value_heads=$HEADS,\
intermediate_size=5462,\
hidden_size=2048,\
num_hidden_layers=24,\
max_position_embeddings=8192,"
SMALL_PARAMS="\
num_attention_heads=$HEADS,\
num_key_value_heads=$HEADS,\
intermediate_size=2730,\
hidden_size=1024,\
num_hidden_layers=24,\
max_position_embeddings=2048,"


# MODEL_PATH=../out/llama_1.4_softmax_32head
# MODEL_PATH=../out/llama_1.4_stickbreaking_notf32_invtemp2.5_32head
MODEL_PATH=../out/llama_sb_350m_lr3e-4_pajama/
MODEL_ARGS="dtype=bfloat16,max_length=4096,pretrained=$MODEL_PATH,parallelize=True"
# MODEL_PARAMS="$LARGE_PARAMS${MODEL_ARGS},class_name=llama_sb_big"
MODEL_PARAMS="$SMALL_PARAMS${MODEL_ARGS},class_name=llama_sb"
export PYTHON_UNBUFFERED=1
lm_eval \
	--model openllama_sb \
	--model_args $MODEL_PARAMS \
	--tasks lambada_openai \
	--batch_size 1 \
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

