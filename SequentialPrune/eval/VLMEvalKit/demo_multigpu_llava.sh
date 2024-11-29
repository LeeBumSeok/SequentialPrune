cd SequentialPrune/eval/VLMEvalKit

#!/bin/bash
set -x
ATTENTION_RANKS=(0.5 0.75 0.9)
AGG_LAYERS=(2 3 5)
for attention_rank in "${ATTENTION_RANKS[@]}"; do
    for agg_layer in "${AGG_LAYERS[@]}"; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 run.py \
            --data DocVQA_VAL OCRBench \
            --model llava_v1.5_7b \
            --work-dir llava_v1.5_7b_rank_${attention_rank}_layer_${agg_layer}_seq \
            --use-fast-v \
            --fast-v-inplace \
            --fast-v-sys-length 36 \
            --fast-v-image-token-length 576 \
            --fast-v-attention-rank ${attention_rank} \
            --fast-v-agg-layer ${agg_layer} \
            --fast-v-sequential-prune
    done
done

