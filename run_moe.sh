#!/bin/bash
set -x
# /proj/checkpoints/shawntan/bucketstuff/out
# accelerate launch -m lm_eval --model sparsegpt \
export PYTHONPATH=../dolomite-engine/
lm_eval --model hf  \
	 --model_args "dtype=bfloat16,pretrained=/proj/checkpoints/shawntan/granitemoe_hf_checkpoint"\
	 --batch_size 1 --tasks wikitext --device cuda --num_fewshot 0
 # /checkpoint/latest.pt/,\
