cd SequentialPrune/eval/VLMEvalKit

#!/bin/bash
export TOKENIZERS_PARALLELISM=false
set -x
ATTENTION_RANKS=(0.5 0.75 0.9 1.0)
AGG_LAYERS=(2 3 5)
for attention_rank in "${ATTENTION_RANKS[@]}"; do
    for agg_layer in "${AGG_LAYERS[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 run.py \
            --data DocVQA_VAL OCRBench \
            --model Qwen2-VL-2B-Instruct \
            --work-dir Qwen2-VL-2B-Instruct_rank_${attention_rank}_layer_${agg_layer} \
            --use-fast-v \
            --fast-v-inplace \
            --fast-v-sys-length 15 \
            --fast-v-attention-rank ${attention_rank} \
            --fast-v-agg-layer ${agg_layer}
    done
done