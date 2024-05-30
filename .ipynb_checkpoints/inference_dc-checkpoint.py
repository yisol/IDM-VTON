# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from ip_adapter.ip_adapter import Resampler

import argparse
import logging
import os
import torch.utils.data as data
import torchvision
import json
import accelerate
import numpy as np
import torch
from PIL import Image, ImageDraw
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer
import cv2
from diffusers.utils.import_utils import is_xformers_available
from numpy.linalg import lstsq

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline



logger = get_logger(__name__, log_level="INFO")

label_map={
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "yisol/IDM-VTON",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="result",)
    parser.add_argument("--category",type=str,default="upper_body",choices=["upper_body", "lower_body", "dresses"])
    parser.add_argument("--unpaired",action="store_true",)
    parser.add_argument("--data_dir",type=str,default="/home/omnious/workspace/yisol/Dataset/zalando")
    parser.add_argument("--seed", type=int, default=42,)
    parser.add_argument("--test_batch_size", type=int, default=2,)
    parser.add_argument("--guidance_scale",type=float,default=2.0,)
    parser.add_argument("--mixed_precision",type=str,default=None,choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    args = parser.parse_args()


    return args

def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class DresscodeTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        category = "upper_body",
        size: Tuple[int, int] = (512, 384),
    ):
        super(DresscodeTestDataset, self).__init__()
        self.dataroot = os.path.join(dataroot_path,category)
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.toTensor = transforms.ToTensor()
        self.order = order
        self.radius = 5
        self.category = category
        im_names = []
        c_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path,category, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path,category, f"{phase}_pairs_{order}.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)


        file_path = os.path.join(dataroot_path,category,"dc_caption.txt")

        self.annotation_pair = {}
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                self.annotation_pair[parts[0]] = ' '.join(parts[1:])


        self.im_names = im_names
        self.c_names = c_names
        self.clip_processor = CLIPImageProcessor()
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = self.category
        cloth = Image.open(os.path.join(self.dataroot, "images", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, "images", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)




        skeleton = Image.open(os.path.join(self.dataroot, 'skeletons', im_name.replace("_0", "_5")))
        skeleton = skeleton.resize((self.width, self.height))
        skeleton = self.transform(skeleton)

        # Label Map
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(os.path.join(self.dataroot, 'label_maps', parse_name))
        im_parse = im_parse.resize((self.width, self.height), Image.NEAREST)
        parse_array = np.array(im_parse)

        # Load pose points
        pose_name = im_name.replace('_0.jpg', '_2.json')
        with open(os.path.join(self.dataroot, 'keypoints', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height / 512.0)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
            point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
            one_map = self.toTensor(one_map)
            pose_map[i] = one_map[0]

        agnostic_mask = self.get_agnostic(parse_array, pose_data, self.category, (self.width,self.height))
        # agnostic_mask = transforms.functional.resize(agnostic_mask, (self.height, self.width),
        #                                              interpolation=transforms.InterpolationMode.NEAREST)

        mask = 1 - agnostic_mask
        im_mask = image * agnostic_mask 
        
        pose_img = Image.open(
            os.path.join(self.dataroot, "image-densepose", im_name)
        )
        pose_img = self.transform(pose_img)  # [-1,1]
 
        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        result["inpaint_mask"] =mask
        result["im_mask"] = im_mask
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["caption"] = "model is wearing a " + cloth_annotation
        result["pose_img"] = pose_img

        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)




    def get_agnostic(self,parse_array, pose_data, category, size):
        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 2).astype(np.float32) + \
                    (parse_array == 3).astype(np.float32) + \
                    (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        if category == 'dresses':
            label_cat = 7
            parse_mask = (parse_array == 7).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif category == 'upper_body':
            label_cat = 4
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif category == 'lower_body':
            label_cat = 6
            parse_mask = (parse_array == 6).astype(np.float32) + \
                        (parse_array == 12).astype(np.float32) + \
                        (parse_array == 13).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                (parse_array == 14).astype(np.float32) + \
                                (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        # dilation
        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        width = size[0]
        height = size[1]

        im_arms = Image.new('L', (width, height))
        arms_draw = ImageDraw.Draw(im_arms)
        if category == 'dresses' or category == 'upper_body':
            shoulder_right = tuple(np.multiply(pose_data[2, :2], height / 512.0))
            shoulder_left = tuple(np.multiply(pose_data[5, :2], height / 512.0))
            elbow_right = tuple(np.multiply(pose_data[3, :2], height / 512.0))
            elbow_left = tuple(np.multiply(pose_data[6, :2], height / 512.0))
            wrist_right = tuple(np.multiply(pose_data[4, :2], height / 512.0))
            wrist_left = tuple(np.multiply(pose_data[7, :2], height / 512.0))
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right], 'white', 30, 'curve')
                else:
                    arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right], 'white', 30,
                                'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line([shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30, 'curve')
                else:
                    arms_draw.line([elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white', 30,
                                'curve')
            else:
                arms_draw.line([wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right], 'white',
                            30, 'curve')

            if height > 512:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((10, 10), np.uint16), iterations=5)
            elif height > 256:
                im_arms = cv2.dilate(np.float32(im_arms), np.ones((5, 5), np.uint16), iterations=5)
            hands = np.logical_and(np.logical_not(im_arms), arms)
            parse_mask += im_arms
            parser_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)
        if category == 'dresses' or category == 'upper_body':
            points = []
            points.append(np.multiply(pose_data[2, :2], height / 512.0))
            points.append(np.multiply(pose_data[5, :2], height / 512.0))
            x_coords, y_coords = zip(*points)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords, rcond=None)[0]
            for i in range(parse_array.shape[1]):
                y = i * m + c
                parse_head_2[int(y - 20 * (height / 512.0)):, i] = 0

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                            np.logical_not(np.array(parse_head_2, dtype=np.uint16))))

        if height > 512:
            parse_mask = cv2.dilate(parse_mask, np.ones((20, 20), np.uint16), iterations=5)
        elif height > 256:
            parse_mask = cv2.dilate(parse_mask, np.ones((10, 10), np.uint16), iterations=5)
        else:
            parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        agnostic_mask = parse_mask_total.unsqueeze(0)
        return agnostic_mask




def main():
    args = parse_args()
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float16
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    #     args.mixed_precision = accelerator.mixed_precision
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16
    #     args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    unet = UNet2DConditionModel.from_pretrained(
        "yisol/IDM-VTON-DC",
        subfolder="unet",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    )
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )


    # Freeze vae and text_encoder and set unet to trainable
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    UNet_Encoder.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    UNet_Encoder.to(accelerator.device, weight_dtype)
    unet.eval()
    UNet_Encoder.eval()

    
    
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    test_dataset = DresscodeTestDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="unpaired" if args.unpaired else "paired",
        category = args.category,
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    pipe = TryonPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            vae=vae,
            feature_extractor= CLIPImageProcessor(),
            text_encoder = text_encoder_one,
            text_encoder_2 = text_encoder_two,
            tokenizer = tokenizer_one,
            tokenizer_2 = tokenizer_two,
            scheduler = noise_scheduler,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
    ).to(accelerator.device)
    pipe.unet_encoder = UNet_Encoder

    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()



    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                for sample in test_dataloader:
                    img_emb_list = []
                    for i in range(sample['cloth'].shape[0]):
                        img_emb_list.append(sample['cloth'][i])
                    
                    prompt = sample["caption"]

                    num_prompts = sample['cloth'].shape[0]                                        
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                    if not isinstance(prompt, List):
                        prompt = [prompt] * num_prompts
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * num_prompts

                    image_embeds = torch.cat(img_emb_list,dim=0)

                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )
                    
                    
                        prompt = sample["caption_cloth"]
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

                        if not isinstance(prompt, List):
                            prompt = [prompt] * num_prompts
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * num_prompts


                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )
                        


                        generator = torch.Generator(pipe.device).manual_seed(args.seed) if args.seed is not None else None
                        images = pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                            num_inference_steps=args.num_inference_steps,
                            generator=generator,
                            strength = 1.0,
                            pose_img = sample['pose_img'],
                            text_embeds_cloth=prompt_embeds_c,
                            cloth = sample["cloth_pure"].to(accelerator.device),
                            mask_image=sample['inpaint_mask'],
                            image=(sample['image']+1.0)/2.0, 
                            height=args.height,
                            width=args.width,
                            guidance_scale=args.guidance_scale,
                            ip_adapter_image = image_embeds,
                        )[0]


                    for i in range(len(images)):
                        x_sample = pil_to_tensor(images[i])
                        torchvision.utils.save_image(x_sample,os.path.join(args.output_dir,sample['im_name'][i]))
                



if __name__ == "__main__":
    main()
