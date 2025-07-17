import torch
from transformers import GPT2Config
import os
import numpy as np
from model import  EmotionInjectionTransformer
from diffusers import StableDiffusionXLPipeline
import torch
from inference import emoticrafter
import matplotlib.pyplot as plt
import argparse
def emoticrafter5x5(pipe,eit, prompt,device = "cuda", seed = 42,save_path = "./results/5x5.png"):
    plt.figure(figsize = (15,15))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    imgs_num = 42
    with torch.no_grad():
        index = 0
        for a in [-3,-1.5,0,1.5,3]:
            for v in [-3,-1.5,0,1.5,3]:
                image = emoticrafter(pipe,eit, prompt,a, v, device = "cuda", seed = seed)
                index += 1
                plt.subplot(5,5, index, aspect='auto')
                plt.axis('off')
                plt.imshow(image)
    plt.savefig(save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type = str)
    parser.add_argument('--ckpt_path', type = str)
    parser.add_argument('--sdxl_path', type = str)
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()
    prompt = args.prompt
    
    device = 'cuda'
    ckpt_path = '/data/emo_generation/emo_generator/dsq/f2f/checkpoints/feature2featureXL-out-Linear+LN-residual-method6-weight-balence/best_model.pth'
    sdxl_path = "/data/emo_generation/emo_generator/local/stable-diffusion-xl-base-1.0"

    config = GPT2Config.from_pretrained('./config')
    eit = EmotionInjectionTransformer(config,final_out_type="Linear+LN").to(device)
    eit = torch.nn.DataParallel(eit)
    ckpt = torch.load(ckpt_path)
    eit.load_state_dict(ckpt)
    eit.eval()
    eit.to(device)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, 
                                                    use_safetensors=True, variant="fp16")
    pipe.to(device)
    
    save_path = f"./results/5x5 | {prompt}.png"
    emoticrafter5x5(pipe,eit,prompt ,seed = args.seed,save_path = save_path)


