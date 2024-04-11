# Improving Diffusion Models for Authentic Virtual Try-on in the Wild
This is an official implementation of paper 'Improving Diffusion Models for Authentic Virtual Try-on in the Wild'
- [paper](https://arxiv.org/abs/2403.05139) 
- [project page](https://idm-vton.github.io/) 

🤗 Try our huggingface [Demo](https://huggingface.co/spaces/yisol/IDM-VTON)

![teaser](assets/teaser.png)&nbsp;


## TODO LIST


- [x] demo model
- [x] inference code
- [ ] training code



## Requirements

```
git clone https://github.com/yisol/IDM-VTON.git
cd IDM-VTON

conda env create -f environment.yaml
conda activate idm
```

## Data preparation

### VITON-HD
You can download VITON-HD dataset from [VITON-HD](https://github.com/shadow2496/VITON-HD).  
After download VITON-HD dataset, move vitonhd_test_tagged.json into the test folder.  
Structure of the Dataset directory should be as follows.

```

train
|-- ...

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json

```

### DressCode
You can download DressCode dataset from [DressCode](https://github.com/aimagelab/dress-code).  
We provide pre-computed densepose images and captions for garments [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/cpis7_kaist_ac_kr/EaIPRG-aiRRIopz9i002FOwBDa-0-BHUKVZ7Ia5yAVVG3A?e=YxkAip).  
We used [detectron2](https://github.com/facebookresearch/detectron2) for obtaining densepose images.  
After download DressCode dataset, place image-densepose directories and caption text files as follows.  

```
DressCode
|-- dresses
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
|-- lower_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
|-- upper_body
    |-- images
    |-- image-densepose
    |-- dc_caption.txt
    |-- ...
```


## Inference


### VITON-HD

Inference using python file with arguments.

```
accelerate launch inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0
```

You can simply run with the script file.

```
sh inference.sh
```

### DressCode

For DressCode dataset, put the category you want to generate images via category argument.
```
accelerate launch inference_dc.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 
    --test_batch_size 2
    --guidance_scale 2.0
    --category "upper_body" 
```

You can simply run with the script file.
```
sh inference.sh
```


## Acknowledgements

For the demo, GPUs are supported from [zerogpu](https://huggingface.co/zero-gpu-explorers), and auto masking generation codes are based on [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) and [DCI-VTON](https://github.com/bcmi/DCI-VTON-Virtual-Try-On).  
Parts of the code are based on [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).



## Citation
```
@article{choi2024improving,
  title={Improving Diffusion Models for Virtual Try-on},
  author={Choi, Yisol and Kwak, Sangkyung and Lee, Kyungmin and Choi, Hyungwon and Shin, Jinwoo},
  journal={arXiv preprint arXiv:2403.05139},
  year={2024}
}
```

## License
The codes and checkpoints in this repository are under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).



