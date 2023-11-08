#!/bin/bash
python main.py \
    --model gpt-st \
    --model_args model_path=$1 \
	--batch_size 32 \
    --tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande,triviaqa,webqs,mathqa,wikitext,lambada_openai,race --no_cache

# boolq,
