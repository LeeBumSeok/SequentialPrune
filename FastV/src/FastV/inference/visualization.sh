# python ./src/FastV/inference/plot_inefficient_attention.py \
#     --model-path "./model/llava-v1.5-7b/"\
#     --image-path "./src/LLaVA/images/llava_logo.png" \
#     --prompt "Describe the image in details."\
#     --output-path "./output_example/except"\

python ./src/FastV/inference/plot_except_fastv_token.py \
    --model-path "./model/llava-v1.5-7b/"\
    --image-path "./src/LLaVA/images/llava_logo.png" \
    --prompt "Describe the image in details."\
    --output-path "./output_example/except"\