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

model_paths=(
	../out/llama_alibi_350m_lr3e-4_pajama/
	../out/llama_fire_350m_lr3e-4_pajama/
	../out/llama_nope_350m_lr3e-4_pajama/
	../out/llama_sb_350m_lr3e-4_pajama/
	../out/llama_sm_350m_lr3e-4_pajama/
	../out/llama_sm_350m_lr3e-4_pajama/
)
exp_names=("alibi" "fire" "nope" "stickbreaking" "rope" "rope_ntk1")
models=("openllama_sb" "openllama_sb" "openllama_sb" "openllama_sb" "openllama" "openllama")
class_names=("llama_alibi" "llama_fire" "llama_nope" "llama_sb" "" "")
ntks=("" "" "" "" "" ",ntk=True")

for config in "${!models[@]}"
do
	MODEL=${models[config]}
	MODEL_PATH=${model_paths[config]}
	CLASS_NAME=${class_names[config]}
	NTK=${ntks[config]}
	echo $MODEL
	echo $MODEL_PATH
	EXP_NAME=${exp_names[config]}
	OUTDIR=./length_extrapolation/$EXP_NAME
	mkdir -p $OUTDIR
	for num_block in 1 2 4 8 16 32
	do
		ctx_length=$((num_block * 2048))
		[[ $CLASS_NAME ]] && CLASS_PARAM="class_name=$CLASS_NAME" || CLASS_PARAM=
		MODEL_ARGS="$CLASS_PARAM,dtype=bfloat16,max_length=$ctx_length,pretrained=$MODEL_PATH,parallelize=True$NTK"
		MODEL_PARAMS=$SMALL_PARAMS$MODEL_ARGS
		lm_eval \
			--model $MODEL \
			--model_args $MODEL_PARAMS\
			--batch_size 1 \
			--tasks wikitext \
			--device cuda --num_fewshot 0 | tee $OUTDIR/$ctx_length.log
	done
	echo ----
done

cd length_extrapolation
for x in *; do echo $x; cd $x; grep word_perplexity *.log | sort -n;  cd ..;done
cd ..
