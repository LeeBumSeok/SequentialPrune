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

contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]


periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + " " in inText or " " + p in inText) or (
            re.search(commaStrip, inText) != None
        ):
            outText = outText.replace(p, "")
        else:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def clean_text(pred):
    pred = pred.replace("\n", " ")
    pred = pred.replace("\t", " ")
    pred = pred.strip()
    pred = processPunctuation(pred)
    pred = processDigitArticle(pred)

    return pred


# %%
def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_dataset(
    image_dir="./data/docvqa/spdocvqa_images",
    ann_path="./data/docvqa/spdocvqa_qas/val_v1.0_withQT.json",
):
    with open(ann_path, "r") as f:
        data = json.load(f)["data"]

    dataset = []
    for item in data:
        question_id = item["questionId"]
        relative_img_path = item["image"]
        corrected_relative_img_path = relative_img_path.replace("documents", "images")
        img_path = os.path.join(image_dir, corrected_relative_img_path)
        question = item["question"]
        answers = item["answers"]
        question_type = item["question_types"]

        dataset.append(
            {
                "question_id": question_id,
                "image_path": img_path,
                "question": question,
                "gt_answers": answers,
                "question_type": question_type
            }
        )

    return dataset


dataset_docvqa_images = []
dataset_docvqa_prompts = []
dataset_docvqa_labels = []

dataset = load_dataset(
    image_dir="data/docvqa/spdocvqa_images",
    ann_path="data/docvqa/spdocvqa_qas/val_v1.0_withQT.json",
)

print(f"Loaded {len(dataset)} data samples.")


# N_examples = 3
N_examples = len(dataset)
cur_num = 0

for i in dataset:
    cur_num += 1
    if cur_num > N_examples:
        break

    dataset_docvqa_images.append(i["image_path"])
    dataset_docvqa_prompts.append(i["question"])
    dataset_docvqa_labels.append(i["gt_answers"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        default="/home/cl/models/llava-v1.5-13b",
    )
    parser.add_argument(
        "--use-fast-v", type=str, required=True, help="whether to use fast-v"
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
    # output path
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="the path to save the output json file",
    )

    pargs = parser.parse_args()

    # %%
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
        image_aspect_ratio = "pad"

    # %%
    args = InferenceArgs()

    # %%

    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    print("Loading model...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )
    print("Model loaded.")

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # set model fastv config
    if pargs.use_fast_v == "True":
        model.config.use_fast_v = True
        model.config.fast_v_sys_length = pargs.fast_v_sys_length
        model.config.fast_v_image_token_length = pargs.fast_v_image_token_length
        model.config.fast_v_attention_rank = pargs.fast_v_attention_rank
        model.config.fast_v_agg_layer = pargs.fast_v_agg_layer
    else:
        model.config.use_fast_v = False

    model.model.reset_fastv()

    # %%
    def inference(prompts, images, metadata):
        outputs = []
        for prompt, image, meta in tqdm(zip(prompts, images, metadata), total=len(prompts)):
            image = load_image(image)
            image = image.convert("RGB")
            image_tensor = process_images([image], image_processor, args)
            conv = conv_templates[args.conv_mode].copy()
            if type(image_tensor) is list:
                image_tensor = [
                    image.to(model.device, dtype=torch.float16)
                    for image in image_tensor
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
            prompt = (
                conv.get_prompt()
            )  # + "In the image, there is a kitchen with a refrigerator, a sink, and a cup on the counter. The cup is placed on the counter, and there is a backpack nearby. The room appears to be empty, and there are no other objects or people in the scene.\n\nTo bring a clean cup to the person, the robot should first look for a cup. Since the cup is already on the counter, the robot can proceed to pick up the cup. After picking up the cup, the robot should then put the cup into the sink to clean it. Finally, the robot can return the clean cup to the person.\n\nBased on the image, the correct sequence of actions for the robot is (A) Look for a cup, (B) Pick up a cup, and (C) Put the cup into the sink."

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

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    attention_mask=None,
                    do_sample=False,
                    max_new_tokens=100,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            output = (
                tokenizer.decode(
                    output_ids["sequences"][0, input_ids.shape[1] :],
                    skip_spectial_tokens=True,
                )
                .strip()
                .replace("</s>", "")
            )
            outputs.append({
            "question_id": meta["question_id"],
            "image_path": meta["image_path"],
            "question": meta["question"],
            "model_answer": output,
            "gt_answers": meta["gt_answers"],
            "question_type": meta["question_type"]
        })
            
        return outputs

    docvqa_inference_outputs = inference(dataset_docvqa_prompts, dataset_docvqa_images, dataset[:N_examples])

    def levenshtein_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(
                        1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                    )
            distances = distances_
        return distances[-1]

    # %%
    # compute ANLS scores
    def evaluate_anls(answer, gt_answers, threshold=0.5):
        '''DOcVQA, InfographicsVQA, STVQA'''
        answer = ' '.join(answer.strip().lower().split())
        if not isinstance(gt_answers, list):
            gt_answers = [gt_answers]
        gt_answers = [' '.join(gt_answer.strip().lower().split()) for gt_answer in gt_answers]

        values = []
        for gt_answer in gt_answers:
            dist = levenshtein_distance(answer, gt_answer)
            length = max(len(answer), len(gt_answer))
            values.append(0.0 if length == 0 else float(dist) / float(length))

        score = 1 - min(values)
        
        score = 0 if score < threshold else score
        
        return score
    
    total_anls_score = 0
    for output in docvqa_inference_outputs:
        per_anls_score = evaluate_anls(output["model_answer"], output["gt_answers"])
        total_anls_score += per_anls_score
        output["anls_score"] = per_anls_score

    average_anls_score = total_anls_score / len(docvqa_inference_outputs)
    
    # Save results to JSON
    output_path = pargs.output_path
    results = {
        "avg": str(average_anls_score),
        "output": docvqa_inference_outputs
    }

    # Write results to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save ANLS score to a separate text file
    anls_score_path = os.path.splitext(output_path)[0] + "_anls_score.txt"
    with open(anls_score_path, "w") as f:
        f.write(f"ANLS Score: {average_anls_score}")