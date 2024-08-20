# %%
import os

# %%
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import json
from tqdm import tqdm
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import os
from datasets import load_from_disk, load_dataset
import torch
import json
from tqdm import tqdm
import re

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm


def visualize_attention(multihead_attention, keep_indices, output_path="atten_map_1.png", title="Layer 5"):
    # Filter attention to only include the kept indices
    filtered_attention = multihead_attention[:, :, keep_indices][:, :, :, keep_indices]
    
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(filtered_attention, axis=1)[0].float()  # Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5), dpi=400)

    # Log normalization
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,
        norm=log_norm,
    )
    
    x_ticks = [str(i*20) for i in range(0, averaged_attention.shape[0])]
    y_ticks = [str(i*20) for i in range(0, averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')

    top_five_attentions = []
    for row in averaged_attention:
        top_values, top_indices = torch.topk(row, 10)
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions, averaged_attention

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str, required=True, default="llava-v1.5-13b")
    parser.add_argument('--image-path', type=str, required=True, help='figs/example.jpg')
    parser.add_argument('--prompt', type=str, required=True, help='describe the image in detail')
    parser.add_argument('--output-path', type=str, required=True, help='the path to save the output json file')
    pargs = parser.parse_args()

    class InferenceArgs:
        model_path = pargs.model_path
        model_base = None
        image_file = None
        device = "cuda"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = 'pad'

    args = InferenceArgs()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    model.config.use_fast_v = True  # Enable FAST_V
    model.model.reset_fastv()
    AGG_LAYER = 5  # Set AGG_LAYER to an appropriate value

    total_layers = model.config.num_hidden_layers

    def inference(prompts, images, append_output=""):
        outputs = []
        outputs_attention = []
        keep_indices_list = []  # To store the keep indices for each layer

        for prompt, image in tqdm(zip(prompts, images), total=len(prompts)):
            image = load_image(image)
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()

            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            inp = prompt

            if image is not None:
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + append_output

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=256,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            output = tokenizer.decode(output_ids['sequences'][0, input_ids.shape[1]:], skip_special_tokens=True).strip().replace("</s>", "")
            outputs.append(output)
            print(output)

            outputs_attention.append(output_ids['attentions'])
            # Calculate keep_indices for each layer based on your logic
            keep_indices = []
            for j in range(total_layers):
                if j < AGG_LAYER:
                    keep_indices.append(torch.arange(input_ids.size(1)))  # Keep all tokens before AGG_LAYER
                else:
                    # Calculate the keep indices based on the attention values
                    last_layer_attention = output_ids['attentions'][0][j]
                    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
                    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
                    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[SYS_LENGTH:SYS_LENGTH+IMAGE_TOKEN_LENGTH]
                    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(ATTENTION_RANK).indices + SYS_LENGTH
                    keep_indices_j = torch.cat( (torch.arange(SYS_LENGTH,device=device), top_attention_rank_index, torch.arange(SYS_LENGTH+IMAGE_TOKEN_LENGTH, seq_length_with_past, device=device)))
                    keep_indices_j = keep_indices_j.sort().values
                    keep_indices.append(keep_indices_j)
            keep_indices_list.append(keep_indices)  # Store the keep indices for each layer

        return outputs, outputs_attention, keep_indices_list

    prompts = [pargs.prompt]
    images = [pargs.image_path]

    model_output_ori, outputs_attention, keep_indices_list = inference(prompts, images)
    model_output, outputs_attention, keep_indices_list = inference(prompts, images, append_output=model_output_ori[0])

    output_path = pargs.output_path

    try:
        os.mkdir(output_path)
    except:
        pass

    try:
        os.mkdir(output_path + "/attn_maps")
    except:
        pass

    with open(output_path + "/output.json", "w") as f:
        json.dump({"prompt": pargs.prompt, "image": pargs.image_path, "output": model_output_ori}, f, indent=4)

    for i, layer_attention in enumerate(outputs_attention):
        for j in range(total_layers):
            keep_indices = keep_indices_list[i][j]  # Get the keep indices for the current layer
            top5_attention, average_attentions = visualize_attention(layer_attention[0][j].cpu(), keep_indices, output_path=output_path + "/attn_maps/atten_map_" + str(j) + ".png", title="Layer " + str(j + 1))