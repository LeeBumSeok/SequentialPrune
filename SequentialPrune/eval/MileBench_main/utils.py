import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

# from icecream import ic
import pandas as pd
from workers.model_workers import LLaVA
from workers.internvl_worker import InternVL
from datasets import load_from_disk, load_dataset

name2worker = {
    "llava-v1.5-7b": LLaVA,
    "InternVL-LLava-7B": InternVL,
    "llava-v1.5-13b": LLaVA,
}
import pandas as pd
MAX_NUM = 200
from tqdm import tqdm


def get_worker_class(name):
    return name2worker[name]


def log_time_to_csv(dataset_name, total_time, output_csv="timing_log.csv"):
    # Check if the CSV file exists
    if os.path.exists(output_csv):
        # Load existing data
        df = pd.read_csv(output_csv)
    else:
        # Create a new DataFrame if the CSV does not exist
        df = pd.DataFrame(columns=["dataset_name", "total_time"])

    # Append new data
    df = df.append(
        {"dataset_name": dataset_name, "total_time": total_time}, ignore_index=True
    )

    # Save the DataFrame back to CSV
    df.to_csv(output_csv, index=False)
    print(f"Logged {dataset_name} with time {total_time} seconds to {output_csv}")


class MileBenchDataset(Dataset):
    def __init__(
        self,
        annotation,
        task_instructions,
        img_dir,
        max_context_len,
        n_tokens_per_image,
        tokenizer,
        dataset_name,
        combine_image=None,
    ):
        """
        Initialize the LongContextBenchmarkDataset class.
        Parameters:
            annotation (list): List of annotations.
            task_instructions (dict): Dictionary of task instructions.
            img_dir (str): Directory containing images.
            max_context_len (int): Maximum number of tokens the model can handle.
            tokenizer: Tokenizer class in Transformers.
            dataset_name: Name of the dataset.
            combine_image (int): Number of combined images.
        """
        self.img_dir = img_dir
        self.annotation = annotation
        self.task_instructions = task_instructions
        self.combine_image = combine_image
        self.max_context_len = max_context_len
        self.n_tokens_per_image = n_tokens_per_image
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

    def __len__(self):
        """
        Get the length of the dataset.
        Returns:
            int: Length of the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Get item by index from the dataset.
        If self.combine_image is not None, set different context prompt.
        Parameters:
            index (int): Index of the item to retrieve.
        Returns:
            dict: Dictionary containing sample information.
        {
            'sample_id': 1,
            'raw_img_list': ['/path/to/image1',],
            'context': 'what is the image <ImageHere> about?',
            'response': '',
        }
        """
        ann = self.annotation[index]

        # Set task instruction
        task_instruction = self.task_instructions[
            ann["task_instruction_id"]
        ]  # at the very beginning
        context = ann["task_instance"]["context"]  # at the very end

        # Set choice_list for multi-choice QA
        if "choice_list" in ann["task_instance"].keys():
            choice_str = "\nChoice list: \n"
            # GPR1200 has more than 26 options which cause bug in answer matching if we use normal alphabetical choices.
            # For this dataset, we simply concatenate all the options with "\n".
            choice_str += "\n".join(
                [
                    (f"{chr(65+idx)}. " if "GPR1200" != self.dataset_name else "")
                    + f"{item}"
                    for idx, item in enumerate(ann["task_instance"]["choice_list"])
                ]
            )
            choice_str += "\nYour answer is: "
            context += choice_str

        # Set prompt
        img_num = len(ann["task_instance"]["images_path"])
        if self.combine_image:
            # set different context prompt for combined images
            for i in range(img_num):
                rmv_txt = "{image#%d}" % (i + 1)
                rmv_tbl = "{table#%d}" % (i + 1)
                context = context.replace(rmv_txt, f"<Image {i+1}> ")
                context = context.replace(rmv_tbl, f"<Image {i+1}> ")
            # context is without instruction here!
            # context = '<ImageHere>'*self.combine_image + '\n' + context # we do this later
        else:
            for i in range(img_num):
                rmv_txt = "{image#%d}" % (i + 1)
                rmv_tbl = "{table#%d}" % (i + 1)
                context = context.replace(rmv_txt, "<ImageHere>")
                context = context.replace(rmv_tbl, "<ImageHere>")

        # Set images paths
        raw_img_list = []
        if self.combine_image:
            combine_image_str = f"combined_{self.combine_image}_images"
            for p in ann["task_instance"][combine_image_str]:
                img_path = os.path.join(
                    self.img_dir.replace(
                        os.path.basename(self.img_dir), combine_image_str
                    ),
                    p,
                )
                raw_img_list.append(img_path)
        else:
            for p in ann["task_instance"]["images_path"]:
                img_path = os.path.join(self.img_dir, p)
                raw_img_list.append(img_path)

        """
        For all datasets, we keep the instruction and question, and truncate the middle part from left.
        """

        image_placeholder = "<ImageHere>"

        ret_img_list = []
        tokenized_instruction = self.tokenizer(
            task_instruction, add_special_tokens=False
        ).input_ids
        tokenized_context = self.tokenizer(context, add_special_tokens=False).input_ids

        # needle datasets
        instruction_length = len(tokenized_instruction)
        length_for_context = self.max_context_len - instruction_length
        # break the question into fragments, then traverse the string in a reverse fashion
        context = context.split(image_placeholder)[::-1]

        past_total_len = 0
        context_id_chunks = []
        ret_img_list = []
        image_start = False  # used for later, deciding we start with text string or image placeholder.

        for fragment in context:
            # add the text first
            cur_ids = self.tokenizer(fragment, add_special_tokens=False).input_ids
            cur_len = len(cur_ids)
            if cur_len + past_total_len > length_for_context:
                if (
                    len(context_id_chunks) == 0
                ):  # if there hasn't been any chunk, we want to truncate a piece from the current text.
                    context_id_chunks.insert(0, cur_ids[-length_for_context:])
                break  # too long!
            image_start = False
            context_id_chunks.insert(0, cur_ids)
            past_total_len += cur_len

            # then concat images
            if not self.combine_image:
                if self.n_tokens_per_image + past_total_len > length_for_context:
                    break  # too long!
                if len(raw_img_list) > 0:
                    image_start = True
                    ret_img_list.insert(0, raw_img_list.pop(-1))
                    past_total_len += self.n_tokens_per_image

        # concat everything together.
        # note that the ending of the input must be text, so we only need to take care of the start.
        # ret_context_str = image_placeholder if image_start else ''
        ret_context_str = ""
        for context_id_chunk in context_id_chunks[:-1]:  # chunks are in correct order
            context_str = self.tokenizer.decode(context_id_chunk)
            ret_context_str += context_str
            ret_context_str += image_placeholder
        ret_context_str += self.tokenizer.decode(
            context_id_chunks[-1]
        )  # add the last chunk without image placeholder appended

        if self.combine_image:
            assert (
                len(raw_img_list) == 1
            ), f"We only support 1 image for combined set, got {len(raw_img_list)} images."
            ret_img_list.insert(
                0, raw_img_list.pop(-1)
            )  # bug for more than 1 image!!!!!
            ret_context_str = (
                image_placeholder + "\n" + task_instruction + "\n" + ret_context_str
            )
            pass
        else:
            if image_start:
                ret_context_str = image_placeholder + ret_context_str
            else:
                pass
            ret_context_str = (
                task_instruction + "\n" + ret_context_str
            )  # prepend task instruction
        # done! We shall return `ret_context_str` and `ret_img_list`
        return {
            "sample_id": ann["sample_id"],
            "context": ret_context_str,
            "raw_img_list": ret_img_list,  # a list of images
            "response": str(ann["response"]),
        }

    def collate_fn(self, batch):
        batch_data = {}
        # Use the default key names
        batch_data["id"] = [sample["sample_id"] for sample in batch]
        batch_data["question"] = [sample["context"] for sample in batch]
        batch_data["image_path"] = [sample["raw_img_list"] for sample in batch]
        batch_data["gt_response"] = [sample["response"] for sample in batch]
        return batch_data


class ANLS_DocVQA_Dataset(Dataset):
    def __init__(
        self,
        annotation,
        img_dir,
        max_context_len,
        n_tokens_per_image,
        tokenizer,
        dataset_name,
    ):
        self.annotation = annotation
        self.img_dir = img_dir
        self.max_context_len = max_context_len
        self.n_tokens_per_image = n_tokens_per_image
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        if MAX_NUM is not None:
            self.annotation = np.load(self.annotation, allow_pickle=True)[1:MAX_NUM]
        else:
            self.annotation = np.load(self.annotation, allow_pickle=True)[1:]
        # ic(len(self.annotation))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        context = ann["question"]  # at the very end
        img_num = len([ann["image_name"]])
        context = "<ImageHere>" + "\n" + context
        # Set images paths
        ret_img_list = [os.path.join(self.img_dir, ann["image_name"] + ".png")]
        answers = list(set(answer.lower() for answer in ann["answers"]))
        return {
            "sample_id": ann["question_id"],
            "context": context,
            "raw_img_list": ret_img_list,  # a list of images
            "response": answers,
        }

    #     {'context': '<ImageHere>
    #    '
    #                'Using the final goal as your guide, reflect on your past '
    #                'successful strategies as a smart agent. Observe the information '
    #                'in the image to inform your present decision.
    #    '
    #                'Your Main Goal:  Place the plate along with the knife and scoop '
    #                'in the right sink.  Step Details: <Image 1> Step#1: Turn left and '
    #                'head towards the sink. <Image 2> Step#2: Pick up the knife to the '
    #                'left of the sink. <Image 3> Step#3: Take a step to the right from '
    #                'where you are at. <Image 4> Step#4: Place the knife on the gold '
    #                'plate with the gold scoop. <Image 5>  Current Step: ',
    #     'raw_img_list': ['../MileBench/ALFRED/combined_1_images/80.jpg'],
    #     'response': 'Pick up the plate along with the knife and scoop.',
    #     'sample_id': 80}

    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.

        Parameters:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched data.
        """
        batch_data = {}
        # Use the default key names
        batch_data["id"] = [sample["sample_id"] for sample in batch]
        batch_data["question"] = [sample["context"] for sample in batch]
        batch_data["image_path"] = [sample["raw_img_list"] for sample in batch]
        batch_data["gt_response"] = [sample["response"] for sample in batch]
        return batch_data


class ROUGE_OCR_VQA_Dataset(Dataset):
    def __init__(
        self,
        annotation,
        img_dir,
        max_context_len,
        n_tokens_per_image,
        tokenizer,
        dataset_name,
    ):
        self.annotation = annotation
        self.img_dir = img_dir
        self.max_context_len = max_context_len
        self.n_tokens_per_image = n_tokens_per_image
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

        self.dataset_ocrvqa_images = []
        self.dataset_ocrvqa_prompts = []
        self.dataset_ocrvqa_labels = []
        self.annotation = load_dataset(
            "/home/work/workspace_bum/Tokenpruning/FastV/data/OCR-VQA"
        )["validation"]
        cur_num = 0
        for i in tqdm(self.annotation):
            cur_num += 1
            if cur_num > MAX_NUM:
                break
            for question, answer in zip(i["questions"], i["answers"]):
                self.dataset_ocrvqa_images.append(i["image"])
                self.dataset_ocrvqa_prompts.append(question)
                self.dataset_ocrvqa_labels.append([answer])


    def __len__(self):
        return len(self.dataset_ocrvqa_images)

    def __getitem__(self, index):
        context = self.dataset_ocrvqa_prompts[index]
        context = "<ImageHere>" + "\n" + context
        # Set images paths
        ret_img_list = [self.dataset_ocrvqa_images[index]]
        answers = self.dataset_ocrvqa_labels[index]
        return {
            "sample_id": int(index),
            "context": context,
            "raw_img_list": ret_img_list,  # a list of images
            "response": answers,
        }

    #     {'context': '<ImageHere>
    #    '
    #                'Using the final goal as your guide, reflect on your past '
    #                'successful strategies as a smart agent. Observe the information '
    #                'in the image to inform your present decision.
    #    '
    #                'Your Main Goal:  Place the plate along with the knife and scoop '
    #                'in the right sink.  Step Details: <Image 1> Step#1: Turn left and '
    #                'head towards the sink. <Image 2> Step#2: Pick up the knife to the '
    #                'left of the sink. <Image 3> Step#3: Take a step to the right from '
    #                'where you are at. <Image 4> Step#4: Place the knife on the gold '
    #                'plate with the gold scoop. <Image 5>  Current Step: ',
    #     'raw_img_list': ['../MileBench/ALFRED/combined_1_images/80.jpg'],
    #     'response': 'Pick up the plate along with the knife and scoop.',
    #     'sample_id': 80}

    def collate_fn(self, batch):
        """
        Custom collate function for batching samples.

        Parameters:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched data.
        """
        batch_data = {}
        # Use the default key names
        batch_data["id"] = [sample["sample_id"] for sample in batch]
        batch_data["question"] = [sample["context"] for sample in batch]
        batch_data["image_path"] = [sample["raw_img_list"] for sample in batch]
        batch_data["gt_response"] = [sample["response"] for sample in batch]
        return batch_data
