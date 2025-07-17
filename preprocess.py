import csv
import random
import torch
import wandb
import argparse
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdxl_path', type = str)
    parser.add_argument('--csv_path', type = str, default = './data/prompt_mapping.csv')
    parser.add_argument('--data_cache_path', type = str, default = "./data/data-cache.pt")
    args = parser.parse_args()
    
    with open(args.csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    device = 'cuda'
    index=0
    sdxl_path = args.sdxl_path
    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(device)
        
    res_list = []
    for item in tqdm(data):
        neural_prompt = item['Neutral_Prompt']
        arousal = float(item['Arousal'])
        valence = float(item['Valence'])
        emotional_prompt = item['Emotional_Prompt']

        with torch.no_grad():
            (   prompt_embeds_ori, 
                negative_prompt_embeds,
                pooled_prompt_embeds_ori, 
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=[neural_prompt ],
                prompt_2=[neural_prompt ],
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
            
            (  prompt_embeds, 
                negative_prompt_embeds,
                pooled_prompt_embeds_ori, 
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=[emotional_prompt],
                prompt_2=[emotional_prompt],
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
                    
        neutral_prompt_feature = prompt_embeds_ori[0]  
            
        emotional_prompt_feature = prompt_embeds[0]
        res =   { 
                'neutral_prompt_feature': neutral_prompt_feature.detach().cpu().to(torch.float16),
                'arousal': torch.tensor([arousal], dtype=torch.float),
                'valence': torch.tensor([valence], dtype=torch.float),
                'emotional_prompt_feature': emotional_prompt_feature.detach().cpu().to(torch.float16)
                }
        index+=1
        res_list.append(res)
    torch.save(res_list,args.data_cache_path)