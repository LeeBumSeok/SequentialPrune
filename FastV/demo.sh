model_path=models/llava-v1.5-7b

python ./demo.py \
        --model-path $model_path \
        --use-fast-v \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank 72 \
        --fast-v-agg-layer 2
