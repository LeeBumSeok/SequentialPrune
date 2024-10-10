#!/bin/bash
export TOKENIZERS_PARALLELISM=false
GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=../../data/MileBench
CUDA_LAUNCH_BLOCKING=0
gpu_num=1

ATTENTION_RANKS=(0.5 0.75 0.9)
AGG_LAYERS=(2 3 5)
for model in llava-v1.5-13b; do
    if [ "$model" == "llava-v1.5-7b" ]; then
        MODEL_CONFIG_PATH=configs/model_configs.yaml
    else
        MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
    fi
    for attention_rank in "${ATTENTION_RANKS[@]}"; do
        for agg_layer in "${AGG_LAYERS[@]}"; do
            python score.py \
                --result-dir outputs_inplace \
                --models ${model}_rank_${attention_rank}_layer_${agg_layer}_seq
        done
    done
done


#!/bin/bash
export TOKENIZERS_PARALLELISM=false
GEN_SCRIPT_PATH=generate.py
EVAL_SCRIPT_PATH=evaluate.py
DATA_DIR=../../data/MileBench
CUDA_LAUNCH_BLOCKING=0
gpu_num=1

ATTENTION_RANKS=(0.5 0.75 0.9)
AGG_LAYERS=(2 3 5)
for model in llava-v1.5-13b; do
    if [ "$model" == "llava-v1.5-7b" ]; then
        MODEL_CONFIG_PATH=configs/model_configs.yaml
    else
        MODEL_CONFIG_PATH=configs/model_configs_13b.yaml
    fi
    for attention_rank in "${ATTENTION_RANKS[@]}"; do
        for agg_layer in "${AGG_LAYERS[@]}"; do
            python score.py \
                --result-dir outputs \
                --models ${model}_rank_${attention_rank}_layer_${agg_layer}
        done
    done
done