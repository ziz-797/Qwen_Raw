
import os
import sys
# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from typing import Optional, Union, Tuple, List, Any, Dict
from transformers import AutoImageProcessor
# def fetch_image(ele: Dict[str, Union[str, Image.Image]], image_patch_size: int = 14) -> Image.Image:
#     if "image" in ele:
#         image = ele["image"]
#     else:
#         image = ele["image_url"]

#     image_obj = None
#     patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)
#     if isinstance(image, Image.Image):
#         image_obj = image
#     elif image.startswith("http://") or image.startswith("https://"):
#         with requests.get(image, stream=True) as response:
#             response.raise_for_status()
#             with BytesIO(response.content) as bio:
#                 image_obj = copy.deepcopy(Image.open(bio))
#     elif image.startswith("file://"):
#         image_obj = Image.open(image[7:])
#     elif image.startswith("data:image"):
#         if "base64," in image:
#             _, base64_data = image.split("base64,", 1)
#             data = base64.b64decode(base64_data)
#             with BytesIO(data) as bio:
#                 image_obj = copy.deepcopy(Image.open(bio))
#     else:
#         image_obj = Image.open(image)
#     if image_obj is None:
#         raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
#     image = to_rgb(image_obj)

#     ## resize
#     if "resized_height" in ele and "resized_width" in ele:
#         resized_height, resized_width = smart_resize(
#             ele["resized_height"],
#             ele["resized_width"],
#             factor=patch_factor,
#         )
#     else:
#         width, height = image.size
#         min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
#         max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
#         resized_height, resized_width = smart_resize(
#             height,
#             width,
#             factor=patch_factor,
#             min_pixels=min_pixels,
#             max_pixels=max_pixels,
#         )
#     image = image.resize((resized_width, resized_height))
#     return image

# def extract_vision_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
#     vision_infos = []
#     if isinstance(conversations[0], dict):
#         conversations = [conversations]
#     for conversation in conversations:
#         for message in conversation:
#             if isinstance(message["content"], list):
#                 for ele in message["content"]:
#                     if (
#                         "image" in ele
#                         or "image_url" in ele
#                         or "video" in ele
#                         or ele.get("type", "text") in ("image", "image_url", "video")
#                     ):
#                         vision_infos.append(ele)
#     return vision_infos


# def process_vision_info(
#     conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
#     return_video_kwargs: bool = False,
#     return_video_metadata: bool = False,
#     image_patch_size: int = 14,
# ) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[torch.Tensor, List[Image.Image]]]], Optional[Dict[str, Any]]]:

#     vision_infos = extract_vision_info(conversations)
#     ## Read images or videos
#     image_inputs = []
#     video_inputs = []
#     video_sample_fps_list = []
#     for vision_info in vision_infos:
#         if "image" in vision_info or "image_url" in vision_info:
#             image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
#         # elif "video" in vision_info:
#         #     video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True,
#         #                 image_patch_size=image_patch_size, return_video_metadata=return_video_metadata)
#         #     video_sample_fps_list.append(video_sample_fps)
#         #     video_inputs.append(video_input)
#         else:
#             raise ValueError("image, image_url or video should in content.")
#     if len(image_inputs) == 0:
#         image_inputs = None
#     if len(video_inputs) == 0:
#         video_inputs = None

#     video_kwargs = {'do_sample_frames': False}
#     if not return_video_metadata: # BC for qwen2.5vl
#         video_kwargs.update({'fps': video_sample_fps_list})

#     if return_video_kwargs:
#         return image_inputs, video_inputs, video_kwargs
#     return image_inputs, video_inputs

class FieldParser:
    def __init__(
            self,
            args,
            tokenizer,
            split
    ): 
        super().__init__()
        self.split = split
        self.args = args
        self.processor = AutoImageProcessor.from_pretrained(args.vision_model)
        self.tokenizer = tokenizer
        self.dataset = args.dataset
        self.patch_base_dir = args.patch_base_dir


    def clean_text(self, text: str) -> str:
        """
        from: https://github.com/X-iZhang/Libra/blob/main/libra/eval/radiology_report.py
        Perform basic cleanup of text by removing newlines, dashes, and some special patterns.
        """
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"\(___, __, __\)", "", text)
        text = re.sub(r"---, ---, ---", "", text)
        text = re.sub(r"\(__, __, ___\)", "", text)
        text = re.sub(r"[_-]+", " ", text)
        text = re.sub(r"[^\w\s.,:;()\-]", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
    def _parse_image(self, image):
        out = self.processor(images=image, return_tensors="pt")
        return out["pixel_values"][0]
    

    def parse(self, features):
        to_return = {'id': str(features['id'])}
        report = features.get("report", "")
        report = self.clean_text(report)  # Clean the report text
        report = report[:512] if self.split == "train" else report
        prompt = features.get("text", "")
        indication = features.get("indication", "")
        comparison = features.get("comparison", "")
        history = features.get("history", "")
        technique = features.get("technique", "")
        full_prompt = prompt.replace("<image>", "")  # qwen3 processor template 会加上 image token
        if self.args.use_clinic_info:
            clinic_info = "Following clinical context:"
            if indication != "":
                clinic_info += f" Indication: {indication}"
            if comparison != "":
                clinic_info += f" Comparison: {comparison}"
            if history != "":
                clinic_info += f" History: {history}"
            if technique != "":
                clinic_info += f" Technique: {technique}"
            full_prompt = full_prompt + " " + clinic_info
        to_return['prompt'] = full_prompt
        to_return['findings'] = report

        # Load image
        cur_img_path = features['image_path'][0]
        image = Image.open(os.path.join(self.args.base_dir, cur_img_path)).convert("RGB")
        image = self._parse_image(image)
        to_return["pixel_values"] = image
        to_return["image_size"] = image.size

        # Prepare messages
        messages = [{"role": "user", "content": full_prompt}]
        if self.split == "train":
            messages.append({"role": "assistant", "content": report})
        
        # Generate text from template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(self.split != "train"))
        # images, videos = process_vision_info(messages, image_patch_size=16)
        # Process inputs
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.args.max_length,   # 你自己设
        )
        
        to_return["input_ids"] = inputs["input_ids"][0]
        to_return["attention_mask"] = inputs["attention_mask"][0]

        # Create labels
        if self.split == "train":
            labels = to_return["input_ids"].clone()
            sep_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            
            # Find sep_ids in input_ids
            # This is a bit slow but works.
            start_idx = -1
            for i in range(len(labels) - len(sep_ids)):
                if labels[i:i+len(sep_ids)].tolist() == sep_ids:
                    start_idx = i + len(sep_ids)
                    break
            
            if start_idx != -1:
                labels[:start_idx] = -100
            else:
                # Fallback or error
                print("Warning: Could not find assistant start token")
                
            to_return["labels"] = labels
        else:
            to_return["labels"] = None

        to_return["padding_side"] = "right" # Qwen usually right padding? Or left for generation?
        # For training right padding is fine if we mask correctly.
        # For generation left padding is usually preferred.
        if self.split != "train":
             to_return["padding_side"] = "left"

        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, tokenizer, split='train'):
        self.args = args
        self.tokenizer = tokenizer
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args, tokenizer=tokenizer, split=split)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args, tokenizer):
    train_dataset = ParseDataset(args, tokenizer, 'train')
    dev_dataset = ParseDataset(args, tokenizer, 'test')
    test_dataset = ParseDataset(args, tokenizer, 'test')
    return train_dataset, dev_dataset, test_dataset

