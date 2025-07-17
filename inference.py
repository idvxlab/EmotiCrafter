import torch
from transformers import GPT2Config
import os
import numpy as np
from model import  EmotionInjectionTransformer
from diffusers import StableDiffusionXLPipeline
import torch
import argparse
def emoticrafter(pipe,eit, prompt,a = 0, v = 0, device = "cuda", seed = 42 ):
    (   prompt_embeds_ori, 
        negative_prompt_embeds,
        pooled_prompt_embeds_ori, 
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    )
    resolution= int(1024)
    out = eit(inputs_embeds = prompt_embeds_ori.to(torch.float32),arousal=torch.FloatTensor([[a]]).to(device),valence=torch.FloatTensor([[v]]).to(device))
    image =pipe(
        prompt_embeds = out[0].to(torch.float16),
        pooled_prompt_embeds =pooled_prompt_embeds_ori,
        guidance_scale=7.5,
        num_inference_steps=25,
        height=resolution,
        width=resolution,
        generator = torch.manual_seed(seed)
    ).images[0]
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type = str)
    parser.add_argument('--arousal', type = float)
    parser.add_argument('--valence', type = float)
    parser.add_argument('--ckpt_path', type = str)
    parser.add_argument('--sdxl_path', type = str)
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()
    
    prompt, arousal, valence = args.prompt, args.arousal, args.valence
    
    device = 'cuda'
    ckpt_path = args.ckpt_path 
    sdxl_path = args.sdxl_path

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
    
    save_path = f"./results/V-A ({valence:.1f},{arousal:.1f}) | {prompt}.png"
    emoticrafter(pipe,eit,prompt , a=arousal, v=valence, seed = args.seed).save(save_path)


