#!/bin/bash
set -x
# /proj/checkpoints/shawntan/bucketstuff/out
# accelerate launch -m lm_eval --model sparsegpt \
OUTDIR=./length_extrapolation/mistral_512/
mkdir -p $OUTDIR
for factor in 1 2 4 8 16 32;
do	
	MAX_LENGTH=$(($factor * 2048))
	lm_eval --model sparsegpt \
	 --model_args "\
class_name=mistral,\
num_attention_heads=32,\
num_key_value_heads=32,\
intermediate_size=2730,\
hidden_size=1024,\
num_hidden_layers=24,\
max_position_embeddings=$MAX_LENGTH,\
dtype=bfloat16,max_length=$MAX_LENGTH,\
pretrained=/proj/checkpoints/shawntan/bucketstuff/out/mistral/"\
	 --batch_size 1 --tasks wikitext --device cuda --num_fewshot 0 | tee $OUTDIR/$MAX_LENGTH.log
 done
 # /checkpoint/latest.pt/,\
