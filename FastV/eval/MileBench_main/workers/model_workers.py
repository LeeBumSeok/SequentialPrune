from workers.baseworker import *
import sys
from PIL import Image
import torch
from icecream import ic
from torch.cuda.amp import autocast
######################## Multi-image application ########################


class LLaVA(BaseWorker):
    def init_components(self, config, args):
        ic(args)
        sys.path.insert(0, "/path/to/LLaVA/packages/")
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
        )

        self.tokenizer, self.model, self.processor, context_len = load_pretrained_model(
            model_path=config.model_dir,
            model_base=None,
            model_name=config.model_dir,
            device="cuda",
        )
        self.use_cache = None
        if args.use_fast_v == True:
            self.use_cache = False
            self.model.config.use_fast_v = True
            self.model.config.fast_v_inplace = args.fast_v_inplace
            self.model.config.fast_v_sys_length = args.fast_v_sys_length
            self.model.config.fast_v_image_token_length = args.fast_v_image_token_length
            self.model.config.fast_v_attention_rank = args.fast_v_attention_rank
            self.model.config.fast_v_agg_layer = args.fast_v_agg_layer
            self.model.config.fast_v_agg_layer = args.fast_v_sequential_prune

        else:
            self.use_cache = True
            self.model.config.use_fast_v = False

        self.model.model.reset_fastv()
        if getattr(self.model.config, "mm_use_im_start_end", False):
            self.single_img_tokens = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
        else:
            self.single_img_tokens = DEFAULT_IMAGE_TOKEN

        self.conv_temp = conv_templates["llava_llama_2"]
        stop_str = (
            self.conv_temp.sep
            if self.conv_temp.sep_style != SeparatorStyle.TWO
            else self.conv_temp.sep2
        )
        self.keywords = [stop_str]
        self.model.cuda()
        self.model.eval()

    def forward(self, questions, image_paths, device, gen_kwargs):
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )

        answers = []
        for question, images_path in zip(questions, image_paths):
            conv = self.conv_temp.copy()
            # Multi-image
            try:
                if images_path == []:
                    image_tensor = None
                else:
                    image_tensor = process_images([
                            Image.open(image_path).convert("RGB")
                            for image_path in images_path
                        ],
                        self.processor,
                        self.model.config,
                    ).to(device)
            except:
                if images_path == []:
                    image_tensor = None
                else:
                    image_tensor = process_images(
                        images_path, self.processor, self.model.config
                    ).to(device)
            question = question.replace(
                "<ImageHere><ImageHere>", "<ImageHere>\n<ImageHere>\n"
            )  # NOTE: handle the special cases in CLEVR-Change dataset
            input_prompt = question.replace("<ImageHere>", self.single_img_tokens)

            conv.append_message(conv.roles[0], input_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(
                    prompt=prompt,
                    tokenizer=self.tokenizer,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                )
                .unsqueeze(0)
                .to(device)
            )
            # Fast V 허깅페이스 버전으로 인한 문제
            with autocast(dtype=torch.bfloat16):
                torch.cuda.empty_cache()
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    use_cache=self.use_cache,
                    stopping_criteria=[
                        KeywordsStoppingCriteria(
                            self.keywords, self.tokenizer, input_ids
                        )
                    ],
                    output_attentions=True,
                    **gen_kwargs
                )

            answer = self.tokenizer.decode(
                output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            answers.append(answer)
            # ic(answer)
        return answers
