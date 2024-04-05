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





## Inference

Inference with python file with argument.

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



## Acknowledgements

For the demo, GPUs are supported from [zerogpu](https://huggingface.co/zero-gpu-explorers), and auto masking generation codes are based on [OOTDiffusion](https://github.com/levihsu/OOTDiffusion).
Parts of the code were based on [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter).
