import torch
import torch.distributed as dist

from vlmeval.config import supported_VLM
from vlmeval.dataset import build_dataset
from vlmeval.inference import infer_data_job
from vlmeval.inference_video import infer_data_job_video
from vlmeval.inference_mt import infer_data_job_mt
from vlmeval.smp import *
from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
import ssl
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context
from icecream import ic
def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
        'none': '0',
        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
    }
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText
import vlmeval.dataset.utils.vqa_eval

def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    # Args that only apply to Video Dataset
    parser.add_argument('--nframe', type=int, default=8)
    parser.add_argument('--pack', action='store_true')
    parser.add_argument('--use-subtitle', action='store_true')
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # API Kwargs, Apply to API VLMs and Judge API LLMs
    parser.add_argument('--nproc', type=int, default=4, help='Parallel API calling')
    parser.add_argument('--retry', type=int, default=None, help='retry numbers for API VLMs')
    # Explicitly Set the Judge Model
    parser.add_argument('--judge', type=str, default=None)
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Configuration for Resume
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')
    # Rerun: will remove all evaluation temp files
    parser.add_argument('--rerun', action='store_true')
    
    
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
    return args

file_name = {'DUDE':'Qwen2-VL-2B-Instruct/Qwen2-VL-2B-Instruct_DUDE.xlsx',
'SLIDEVQA_MINI':'Qwen2-VL-2B-Instruct/Qwen2-VL-2B-Instruct_SLIDEVQA_MINI.xlsx'}

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
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values

def process_answer(answer):
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer

def main():
    args = parse_args()
    
    result_dict = {"SLIDEVQA_MINI":0,"DUDE":0}
    for data in args.data:
        eval_file_path = os.path.join(args.work_dir, file_name[data])
        
        df = pd.read_excel(eval_file_path)
        print(df)
        anls_list = []
        for index, row in df.iterrows():
            # print(row)  # 각 행을 출력
            answer = str(row['answer'])
            prediction = str(row['prediction'])
            answer = process_answer(answer)
            prediction = process_answer(prediction)
            anls_value = anls_compute(answer,prediction)
            anls_value = 1.0 - anls_value
            anls_value = anls_value if anls_value >= 0.5 else 0
            anls_list.append(anls_value)
            # print(answer, prediction, anls_value)/
            # breakpoint()
        anls_list = np.array(anls_list)
        final_score = np.mean(anls_list)
        result_dict[data] = final_score
    print(args.work_dir)
    print(result_dict)
        
if __name__ == '__main__':
    # load_env()
    main()
