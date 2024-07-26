#!/bin/bash

export PYTHON_UNBUFFERED=1
lm_eval \
    --model mixtral_scattermoe \
    --model_args \
pretrained=../scattermoe/examples/converted,\
dtype=bfloat16,max_length=2048,\
parallelize=True,\
    --batch_size 32 \
    --tasks openbookqa,piqa,sciq,arc_easy,arc_challenge,boolq,copa,hellaswag,winogrande,race,wikitext \
    --device cuda --num_fewshot 0  2>&1

# pretrained=mistralai/Mixtral-8x7B-v0.1,\
