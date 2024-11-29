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
from icecream import ic
import csv
from memory_profiler import memory_usage
warnings.simplefilter(action="ignore", category=FutureWarning)
ANLS_DOCVQA_IMG_DIR = (
    "/home/work/workspace_bum/Tokenpruning/FastV/data/docvqa/spdocvqa_images"
)
ANLS_DOCVQA_ANN = (
    "/home/work/workspace_bum/Tokenpruning/FastV/data/docvqa/spdocvqa_imdb/imdb_val.npy"
)
ROUGE_OCRVQA_ANN = "/workspace/eval/OCR-VQA"


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

    # Fastv
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
        type=float,
        required=False,
        help="the rank of attention matrix",
    )
    parser.add_argument(
        "--fast-v-agg-layer",
        type=int,
        required=False,
        help="the layer of attention matrix",
    )
    parser.add_argument(
        "--fast-v-sequential-prune",
        default=False,
        action="store_true",
        help="sequential_prune",
    )
    parser.add_argument(
        "--prune-step",
        type=int,
        required=False,
        default=10,
        help="prune_step_size",
    )
    args = parser.parse_args()
    if args.use_fast_v:
        args.output_pth = os.path.join(
            args.output_dir, f"{args.dataset_name}/pred.json"
        )
    else:
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

        with open(args.output_pth, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(filtered_results)} results to {args.output_pth}")
    else:
        if os.path.exists(args.output_pth):
            if not args.overwrite:
                print(
                    f"{args.output_pth} exists. Please pass `overwrite=True` to avoid unwanted overwriting."
                )
                exit(0)

        with open(args.output_pth, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


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
    worker_class = get_worker_class(args.model_name)
    models_configs = OmegaConf.load(args.model_configs)
    if dataset_name == "ANLS_DocVQA":
        models_configs[args.model_name]["gen_kwargs"]["max_new_tokens"] = 100
    elif dataset_name == "Rouge_OCR_VQA":
        models_configs[args.model_name]["gen_kwargs"]["max_new_tokens"] = 50

    if not models_configs.get(args.model_name):
        raise ValueError
    config = models_configs[args.model_name]
    config.device = device
    worker = worker_class.from_config(config=config, args=args)
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
                tokenizer=worker.tokenizer,
                dataset_name=dataset_name,
            )
        elif dataset_name == "Rouge_OCR_VQA":
            lc_dataset = ROUGE_OCR_VQA_Dataset(
                annotation=ROUGE_OCRVQA_ANN,
                img_dir=ROUGE_OCRVQA_ANN,
                max_context_len=config.max_context_len,
                n_tokens_per_image=config.n_tokens_per_image,
                tokenizer=worker.tokenizer,
                dataset_name=dataset_name,
            )
        lc_dataloader = DataLoader(
            dataset=lc_dataset,
            batch_size=args.batch_image,
            shuffle=False,
            num_workers=8,
            collate_fn=lc_dataset.collate_fn,
            pin_memory=True,
        )
        # dataloader 반복문
        total_batch_time = 0
        for batch_idx, batch in enumerate(tqdm(lc_dataloader)):
            batch_start_time = time.time()  # Start time for the batch
            outputs = worker(device=device, **batch)  # list[dict], with the key "answer" added to each item
            batch_end_time = time.time()  # End time for the batch
            
            batch_time = batch_end_time - batch_start_time
            total_batch_time += batch_time
            
            all_predictions = outputs
            prediction_results.extend(all_predictions)
        print(f"Total time elapsed: {total_batch_time} seconds")

        # Define CSV file path
        csv_file = 'execution_times.csv'

        # Check if CSV file exists
        file_exists = os.path.isfile(csv_file)

        # Open the CSV file in append mode and write the elapsed time
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If file doesn't exist, write the header first
            if not file_exists:
                writer.writerow(["name", "Elapsed Time (seconds)"])
            
            # Write the batch index and elapsed time
            writer.writerow([args.output_dir, total_batch_time])

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
                tokenizer=worker.tokenizer,
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
            for batch in tqdm(lc_dataloader):
                outputs = worker(device=device, **batch)  # list[dict], with the key "answer" added to each item
                all_predictions = outputs
                prediction_results.extend(all_predictions)
            # remove the repetition
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
