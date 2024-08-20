export CUDA_VISIBLE_DEVICES=3

model_path=./model/llava-v1.5-7b
output_path=./results/ocrvqa_eval_fastv
mkdir -p $output_path

rank_list=(144 72 288 432) # rank equals to (1-R)*N_Image_Tokens, R=(75% 50% 25% 12.5%)
Ks=(2) 

for rank in ${rank_list[@]}; do
    for k in ${Ks[@]}; do
    # auto download the ocrvqa dataset
    python ./src/FastV/inference/eval/inference_ocrvqa.py \
        --model-path $model_path \
        --use-fast-v True \
        --fast-v-sys-length 36 \
        --fast-v-image-token-length 576 \
        --fast-v-attention-rank $rank \
        --fast-v-agg-layer $k \
        --output-path $output_path/ocrvqa_7b_FASTV_nocache_mt40_${rank}_${k}.json 
    done
done
