import os

import torch
from diffusers import (AutoencoderKL, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from omegaconf import OmegaConf

from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit
from easyanimate.models.transformer2d import Transformer2DModel
from easyanimate.pipeline.pipeline_pixart_magvit import PixArtAlphaMagvitPipeline
from easyanimate.utils.lora_utils import merge_lora

# Config and model path
config_path         = "config/easyanimate_image_normal_v1.yaml"
#config_path         = "config/easyanimate_video_slicevae_motion_module_v3.yaml"
model_name          = "models/Diffusion_Transformer/PixArt-XL-2-512x512"
#model_name          = "models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512"
sampler_name        = "DPM++"

# Load pretrained model if need
transformer_path    = '/home/lgk/2024/EasyAnimate-main/models/Diffusion_Transformer/PixArt-XL-2-512x512/transformer/diffusion_pytorch_model.safetensors'
#transformer_path    = '/home/lgk/2024/EasyAnimate-main/models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512/transformer/diffusion_pytorch_model.safetensors'
#vae_path            = '/home/lgk/2024/EasyAnimate-main/models/Diffusion_Transformer/EasyAnimateV3-XL-2-InP-512x512/vae/diffusion_pytorch_model.safetensors'
vae_path            = '/home/lgk/2024/EasyAnimate-main/models/Diffusion_Transformer/PixArt-XL-2-512x512/vae/diffusion_pytorch_model.safetensors'
#lora_path           = '/home/lgk/2024/EasyAnimate-main/models/easyanimate_portrait_lora.safetensors'
#lora_path           = '/home/lgk/2024/EasyAnimate-main/models/Personalized_Model/checkpoint-300.safetensors'
lora_path           = '/home/lgk/2024/EasyAnimate-main/models/checkpoint-6000.safetensors'
# Other params
sample_size     = [512, 512]
weight_dtype    = torch.bfloat16
prompt          = "A high-angle, top-down view of several large, industrial bulk bags (FIBC bags) piled together in a dimly lit warehouse or storage area. The image is rendered in monochrome (black and white), emphasizing texture and shadow. The bags are made of heavy-duty woven polypropylene fabric, showing deep wrinkles, folds, and creases from being filled and handled. Thick, sturdy webbing straps are visible, crisscrossing over the tops of the bags, some tied or looped loosely. The lighting is dramatic and directional, creating strong highlights on the raised surfaces of the fabric and deep, dark shadows in the crevices between the bags. The overall mood is gritty, industrial, and slightly desaturated, with a shallow depth of field that keeps the central bags in sharp focus while the periphery fades into darkness. The composition is dense and cluttered, suggesting a working environment."
#prompt          = "1man, bangs, blue eyes, blunt bangs, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt."
negative_prompt = "bad detailed"
guidance_scale  = 6.0
seed            = 43
num_images      = 40  # 设置要生成的图像数量
lora_weight     = 0.55
save_path       = "samples/easyanimate-images"

config = OmegaConf.load(config_path)

# Get Transformer
transformer = Transformer2DModel.from_pretrained(
    model_name, 
    subfolder="transformer"
).to(weight_dtype)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
if OmegaConf.to_container(config['vae_kwargs'])['enable_magvit']:
    Choosen_AutoencoderKL = AutoencoderKLMagvit
else:
    Choosen_AutoencoderKL = AutoencoderKL
vae = Choosen_AutoencoderKL.from_pretrained(
    model_name, 
    subfolder="vae", 
    torch_dtype=weight_dtype
)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
    assert len(u) == 0

# Get Scheduler
Choosen_Scheduler = {
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "DPM++": DPMSolverMultistepScheduler, 
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
}[sampler_name]
scheduler = Choosen_Scheduler(**OmegaConf.to_container(config['noise_scheduler_kwargs']))

pipeline = PixArtAlphaMagvitPipeline.from_pretrained(
    model_name,
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=weight_dtype
)
pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight)

# 为每张图生成不同的种子（可选：也可固定种子以复现）
generators = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(num_images)]

with torch.no_grad():
    samples = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        height=sample_size[0],
        width=sample_size[1],
        num_images_per_prompt=num_images,
        generator=generators,  # 或者传入单个 generator（会自动广播），但显式传列表更稳妥
    ).images  # now a list of PIL images

# Save
os.makedirs(save_path, exist_ok=True)
existing_files = [f for f in os.listdir(save_path) if f.endswith('.png')]
start_index = len(existing_files) + 1

for i, image in enumerate(samples):
    prefix = str(start_index + i).zfill(8)
    image_path = os.path.join(save_path, prefix + ".png")
    image.save(image_path)
