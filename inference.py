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
from PIL import Image
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, StableDiffusionXLControlNetInpaintPipeline
from transformers import AutoTokenizer, PretrainedConfig,CLIPImageProcessor, CLIPVisionModelWithProjection,CLIPTextModelWithProjection, CLIPTextModel, CLIPTokenizer

from diffusers.utils.import_utils import is_xformers_available

from src.unet_hacked_tryon import UNet2DConditionModel
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline



logger = get_logger(__name__, log_level="INFO")



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path",type=str,default= "yisol/IDM-VTON",required=False,)
    parser.add_argument("--width",type=int,default=768,)
    parser.add_argument("--height",type=int,default=1024,)
    parser.add_argument("--num_inference_steps",type=int,default=30,)
    parser.add_argument("--output_dir",type=str,default="result",)
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

#Using a pretrained inception model to evaluate the GAN with the inception score
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batch_size_i = batch.size()[0]
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class VitonHDTestDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDTestDataset, self).__init__()
        self.dataroot = dataroot_path
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

        with open(
            os.path.join(dataroot_path, phase, "vitonhd_" + phase + "_tagged.json"), "r"
        ) as file1:
            data1 = json.load(file1)

        annotation_list = [
            "sleeveLength",
            "neckLine",
            "item",
        ]

        self.annotation_pair = {}
        for k, v in data1.items():
            for elem in v:
                annotation_str = ""
                for template in annotation_list:
                    for tag in elem["tag_info"]:
                        if (
                            tag["tag_name"] == template
                            and tag["tag_category"] is not None
                        ):
                            annotation_str += tag["tag_category"]
                            annotation_str += " "
                self.annotation_pair[elem["file_name"]] = annotation_str

        self.order = order
        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ = line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.clip_processor = CLIPImageProcessor()
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        if c_name in self.annotation_pair:
            cloth_annotation = self.annotation_pair[c_name]
        else:
            cloth_annotation = "shirts"
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name)
        ).resize((self.width,self.height))
        image = self.transform(im_pil_big)

        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-mask", im_name.replace('.jpg','_mask.png'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
        mask = 1-mask
        im_mask = image * mask
 
        pose_img = Image.open(
            os.path.join(self.dataroot, self.phase, "image-densepose", im_name)
        )
        pose_img = self.transform(pose_img)  # [-1,1]
 
        result = {}
        result["c_name"] = c_name
        result["im_name"] = im_name
        result["image"] = image
        result["cloth_pure"] = self.transform(cloth)
        result["cloth"] = self.clip_processor(images=cloth, return_tensors="pt").pixel_values
        result["inpaint_mask"] =1-mask
        result["im_mask"] = im_mask
        result["caption_cloth"] = "a photo of " + cloth_annotation
        result["caption"] = "model is wearing a " + cloth_annotation
        result["pose_img"] = pose_img

        return result

    def __len__(self):
        # model images + cloth image
        return len(self.im_names)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is the generator output
            nn.Linear(generator_output_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            # Output is a single value: real or fake
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

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
        args.pretrained_model_name_or_path,
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

    test_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir,
        phase="test",
        order="unpaired" if args.unpaired else "paired",
        size=(args.height, args.width),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=4,
    )

    #Training Set
    train_dataset = VitonHDTestDataset(
        dataroot_path=args.data_dir,
        phase="train",
        order="unpaired" if args.unpaired else "paired",
        size=(args.height, args.width),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=32,
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

                        # Define an optimizer for the generator
                        optimizer_g = torch.optim.Adam(generator, lr=args.lr)


                        # Define a loss function
                        criterion = torch.nn.BCELoss()

                        # Creating Discriminator for further training the generator
                        discriminator = Discriminator().to(pipe.device)
                        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

                        # Training loop
                        for epoch in range(args.num_epochs):
                            for i, data in enumerate(train_dataloader, 0):
                                # 1. Train the discriminator
                                discriminator.zero_grad()
                                
                                # 1a. Train the discriminator on real data
                                real_data = data.to(pipe.device)
                                real_output = discriminator(real_data)
                                real_loss = criterion(real_output, torch.ones_like(real_output))
                                real_loss.backward()

                                # 1b. Train the discriminator on fake data
                                noise = torch.randn(32, 100, 1, 1, device=pipe.device)
                                if generator:
                                    fake_data = generator(noise)
                                    fake_output = discriminator(fake_data.detach())
                                    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
                                    fake_loss.backward()

                                d_loss = real_loss + fake_loss
                                optimizer_d.step()

                                # 2. Train the generator
                                if generator:
                                    generator.zero_grad()

                                # Generate fake data
                                noise = torch.randn(32, 100, 1, 1, device=pipe.device)
                                if generator:
                                    fake_data = generator(noise)

                                # Try to fool the discriminator
                                output = discriminator(fake_data)
                                g_loss = criterion(output, torch.ones_like(output))

                                # Backward pass and optimization
                                g_loss.backward()
                                optimizer_g.step()
                    

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

                    # Calculate the inception score
                    mean, std = inception_score(fake_data)
                    print('Inception score: mean = {}, std = {}'.format(mean, std))


                    for i in range(len(images)):
                        x_sample = pil_to_tensor(images[i])
                        torchvision.utils.save_image(x_sample,os.path.join(args.output_dir,sample['im_name'][i]))
                



if __name__ == "__main__":
    main()
