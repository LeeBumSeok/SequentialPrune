from datetime import datetime
from argparse import ArgumentParser
from accelerate import Accelerator
import json, os
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from utils import (
    get_worker_class,
    MileBenchDataset,
    ANLS_DocVQA_Dataset,
    ROUGE_OCR_VQA_Dataset,
)
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import gc
import warnings

import requests
from PIL import Image
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from matplotlib.colors import LogNorm
from io import BytesIO

from icecream import ic

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

warnings.simplefilter(action="ignore", category=FutureWarning)  #
ANLS_DOCVQA_IMG_DIR = (
    "/home/work/workspace_bum/Tokenpruning/FastV/data/docvqa/spdocvqa_images"
)
ANLS_DOCVQA_ANN = (
    "/home/work/workspace_bum/Tokenpruning/FastV/data/docvqa/spdocvqa_imdb/imdb_val.npy"
)
ROUGE_OCRVQA_ANN = "/workspace/eval/OCR-VQA"


def visualize_attention(
    multihead_attention, title="Layer 5", sample_style="All layers"
):
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()
    averaged_attention = (
        torch.nn.functional.avg_pool2d(
            averaged_attention.unsqueeze(0).unsqueeze(0), 20, stride=20
        )
        .squeeze(0)
        .squeeze(0)
    )

    cmap = plt.cm.get_cmap("viridis")

    plt.figure(figsize=(5, 5), dpi=400)
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    ax = sns.heatmap(averaged_attention, cmap=cmap, norm=log_norm)

    x_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[0])]
    y_ticks = [str(i * 20) for i in range(0, averaged_attention.shape[0])]
    ax.set_xticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_yticks([i for i in range(0, averaged_attention.shape[0])])
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)

    # label ticks
    for label in ax.get_xticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color("blue")
            label.set_fontweight("bold")
        elif 40 < tick_location <= 600:
            label.set_color("red")

    for label in ax.get_yticklabels():
        tick_location = int(label.get_text())
        if 0 <= tick_location <= 40:
            # set the color of the tick labels
            label.set_color("blue")
            label.set_fontweight("bold")
        elif 40 < tick_location <= 600:
            label.set_color("red")

    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.title(title, fontsize=20)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).copy()
    if sample_style == "All layers":
        image = image.resize((768, 768))
    else:
        image = image.resize((1024, 1024))
    buf.close()
    plt.close()

    return image


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def concatenate_images(images_list, number_rows=5, number_cols=7):
    assert len(images_list) == number_rows * number_cols

    # Assuming all images are the same size
    img_width, img_height = images_list[0].size

    # Creating a blank canvas for the final image
    final_img = Image.new("RGB", (img_width * number_cols, img_height * number_rows))

    # Loop over the images and paste them onto the canvas
    for idx, img in enumerate(images_list):
        row = idx // number_cols  # row index
        col = idx % number_cols  # column index

        # paste the image at the correct position on the canvas
        final_img.paste(img, (img_width * col, img_height * row))

    return final_img


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="data/MileBench")
    parser.add_argument("--dataset_name", default="data/sample.json")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--bsz", default=1, type=int)
    parser.add_argument("--batch-image", default=1, type=int)
    parser.add_argument(
        "--combine_image",
        default=None,
        type=int,
        help="Use combined N images for evaluation.",
    )
    parser.add_argument("--model_configs", default="configs/model_configs.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        default="/home/work/workspace_bum/Tokenpruning/FastV/models/llava-v1.5-7b",
    )
    parser.add_argument(
        "--use-fast-v", default=False, action="store_true", help="whether to use fast-v"
    )
    parser.add_argument(
        "--fast-v-inplace",
        default=False,
        action="store_true",
        help="whether to use fast-v inplace version to check real latency, no supported kv cache yet",
    )
    parser.add_argument(
        "--fast-v-sys-length",
        type=int,
        required=False,
        help="the length of system prompt",
    )
    parser.add_argument(
        "--fast-v-image-token-length",
        type=int,
        required=False,
        help="the length of image token",
    )
    parser.add_argument(
        "--fast-v-attention-rank",
        type=int,
        required=False,
        help="the rank of attention matrix",
    )
    parser.add_argument(
        "--fast-v-agg-layer",
        type=int,
        required=False,
        help="the layer of attention matrix",
    )

    args = parser.parse_args()
    args.output_pth = os.path.join(
        args.output_dir, f"{args.model_name}/{args.dataset_name}/pred.json"
    )

    os.makedirs(os.path.dirname(args.output_pth), exist_ok=True)
    return args


def split_data(data):
    """
    Split the data by the images number
    ex: {
        2: [sample1, ...]
        3: [sample2, ...]
    }
    """
    data_dict = {}
    for d in data:
        n_img = len(d["task_instance"]["images_path"])
        if n_img in data_dict:
            data_dict[n_img].append(d)
        else:
            data_dict[n_img] = [d]
    return data_dict


def save(results, args):
    if args.dataset_name == "Rouge_OCR_VQA":
        filtered_results = []
        for result in results:
            result["image"] = ["skip"]
            filtered_results.append(result)
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(
                    f"{args.output_pth} exists. Please pass `--overwrite` to avoid unwanted overwriting."
                )
                exit(0)

        json.dump(
            filtered_results, open(args.output_pth, "w"), ensure_ascii=False, indent=4
        )
        print(f"Saved {len(filtered_results)} results to {args.output_pth}")
    else:
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(
                    f"{args.output_pth} exists. Please pass `overwrite=True` to avoid unwanted overwriting."
                )
                exit(0)
        json.dump(results, open(args.output_pth, "w"), ensure_ascii=False, indent=4)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def temp_inference(
    prompts, images, model, tokenizer, image_processor, conv_param, append_output=None
):
    outputs = []
    outputs_attention = []
    if append_output is None:
        append_output_str = ""
    else:
        append_output_str = append_output
    for prompt, image in zip(prompts, images):
        image_tensor = process_images([image], image_processor, args)
        conv = conv_templates[conv_param].copy()
        if type(image_tensor) is list:
            image_tensor = [
                image.to(model.device, dtype=torch.float16) for image in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp = prompt

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + inp
                )  # False
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + append_output_str

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        image_token_indices = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)

        image_index = []

        for batch_index, seq_index in zip(*image_token_indices):
            image_index.append(seq_index.item())

        model.config.image_token_index = image_index
        model.model.load_image_index()

        with torch.inference_mode():
            start = time.time()
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
            time_cost = time.time() - start

        output = (
            tokenizer.decode(
                output_ids["sequences"][0, input_ids.shape[1] :],
                skip_spectial_tokens=True,
            )
            .strip()
            .replace("</s>", "")
        )
        outputs.append(output)

        outputs_attention.append(output_ids["attentions"])
        if len(outputs) > 1:
            print(output)
        if append_output is None:
            return outputs, outputs_attention, time_cost
    return outputs, outputs_attention


def main(args):
    import torch.distributed as dist

    # accelerator = Accelerator()
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.bsz
    # accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.bsz * dist.get_world_size()
    # accelerator.print(f'{datetime.now()}: Generation of {args.model_name} to {args.dataset_name}')

    ######################### Loading Data #########################
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    combine_image = args.combine_image
    dataset_dir = os.path.join(data_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, "images")
    model_name = args.model_name
    device = "cuda"

    if (
        dataset_name == "ANLS_DocVQA" or dataset_name == "Rouge_OCR_VQA"
    ):  # 귀찮아서 대충
        pass
    else:
        core_annotation = json.load(
            open(
                os.path.join(
                    dataset_dir,
                    (
                        f"{dataset_name}_combined_{combine_image}.json"
                        if combine_image and combine_image != 1
                        else f"{dataset_name}.json"
                    ),
                )
            )
        )
        # split data by images number
        data_dict = split_data(core_annotation["data"])
    ################################################################

    #################### Initializing Worker ######################
    class InferenceArgs:
        model_path = args.model_path
        model_base = None
        image_file = None
        device = "cuda:0"
        conv_mode = None
        temperature = 0.2
        max_new_tokens = 512
        load_8bit = False
        load_4bit = False
        debug = False
        image_aspect_ratio = "pad"

    iargs = InferenceArgs()

    models_configs = OmegaConf.load(args.model_configs)

    if dataset_name == "ANLS_DocVQA":
        models_configs[args.model_name]["gen_kwargs"]["max_new_tokens"] = 100
    elif dataset_name == "Rouge_OCR_VQA":
        models_configs[args.model_name]["gen_kwargs"]["max_new_tokens"] = 50

    if not models_configs.get(args.model_name):
        raise ValueError
    config = models_configs[args.model_name]

    config.device = device

    model_name = get_model_name_from_path(args.model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_name=args.model_name,
        model_base=None,
    )
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if iargs.conv_mode is not None and conv_mode != iargs.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, iargs.conv_mode, iargs.conv_mode
            )
        )
    else:
        iargs.conv_mode = conv_mode

    if args.use_fast_v == True:
        model.config.use_fast_v = True
        model.config.fast_v_inplace = args.fast_v_inplace
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
    else:
        model.config.use_fast_v = False

    model.model.reset_fastv()

    # prepare model for accelerator
    # worker.model = accelerator.prepare(worker.model)

    ################################################################
    ###################### Start Generating ########################
    if dataset_name == "ANLS_DocVQA" or dataset_name == "Rouge_OCR_VQA":
        print("Initialization Finished")
        print(f"Predicting {dataset_name} Using {model_name}")
        prediction_results = []
        if dataset_name == "ANLS_DocVQA":
            lc_dataset = ANLS_DocVQA_Dataset(
                annotation=ANLS_DOCVQA_ANN,
                img_dir=ANLS_DOCVQA_IMG_DIR,
                max_context_len=config.max_context_len,
                n_tokens_per_image=config.n_tokens_per_image,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
            )
        elif dataset_name == "Rouge_OCR_VQA":
            lc_dataset = ROUGE_OCR_VQA_Dataset(
                annotation=ROUGE_OCRVQA_ANN,
                img_dir=ROUGE_OCRVQA_ANN,
                max_context_len=config.max_context_len,
                n_tokens_per_image=config.n_tokens_per_image,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
            )
        lc_dataloader = DataLoader(
            dataset=lc_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=lc_dataset.collate_fn,
            pin_memory=True,
        )
        # lc_dataloader = accelerator.prepare_data_loader(lc_dataloader, device_placement=True)

        # dataloader 반복문
        for batch_idx, batch in enumerate(tqdm(lc_dataloader)):
            # 데이터로더에서 데이터 꺼내는 시간 측정
            # output 구하는 시간 측정
            outputs = model(device=device, **batch)
            all_predictions = outputs
            prediction_results.extend(all_predictions)

        # remove the repetition
        prediction_results = list(
            {item["sample_id"]: item for item in prediction_results}.values()
        )
        print(f"Generation done {len(prediction_results)}")
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("Initialization Finished")
        print(f"Predicting {dataset_name} Using {model_name}")
        prediction_results = []
        for n_img, sub_data in data_dict.items():
            print(f"Proceeding {n_img}-length images samples | Num: {len(sub_data)}")
            lc_dataset = MileBenchDataset(
                annotation=sub_data,
                task_instructions=core_annotation["meta_data"]["task_instruction"],
                img_dir=img_dir,
                max_context_len=config.max_context_len,
                n_tokens_per_image=config.n_tokens_per_image,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                combine_image=combine_image,
            )
            lc_dataloader = DataLoader(
                dataset=lc_dataset,
                batch_size=max(int(args.batch_image / n_img), 1),
                shuffle=False,
                num_workers=8,
                collate_fn=lc_dataset.collate_fn,
                pin_memory=True,
            )
            # lc_dataloader = accelerator.prepare_data_loader(lc_dataloader, device_placement=True)

            for batch in tqdm(lc_dataloader):
                prompts = batch["question"]
                images = batch["image_path"][0]

                convert_image_list = []

                for image in images:
                    convert_image_list.append(load_image(image))

                outputs, outputs_attention, time_cost = temp_inference(
                    prompts,
                    convert_image_list,
                    model,
                    tokenizer,
                    image_processor,
                    iargs.conv_mode,
                )
                all_predictions = outputs
                prediction_results.extend(all_predictions)
            # remove the repetition
            print(prediction_results)
            prediction_results = list(
                {item["sample_id"]: item for item in prediction_results}.values()
            )
            print(f"Generation done {len(prediction_results)}")
            gc.collect()
            torch.cuda.empty_cache()

    save(prediction_results, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done gen\n\n")
