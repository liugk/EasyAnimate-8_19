import os
from diffusers import DiffusionPipeline
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
from safetensors.torch import load_file
from diffusers import PixArtAlphaPipeline
def load_lora_into_transformer(transformer, lora_path, alpha=1.0):
    state_dict = load_file(lora_path)
    # 构建模块名到模块的映射
    named_modules = dict(transformer.named_modules())
    
    for key in state_dict.keys():
        if "lora_down" in key:
            # 例如: transformer.transformer_blocks.0.attn1.to_q.lora_down.weight
            prefix = ".".join(key.split(".")[:-2])  # 去掉 .lora_down.weight
            down_weight = state_dict[key]
            up_weight = state_dict[key.replace("lora_down", "lora_up")]
            
            if prefix in named_modules:
                module = named_modules[prefix]
                # 注入 LoRA
                with torch.no_grad():
                    # 假设是 Linear 层
                    if hasattr(module, "weight"):
                        # 计算 delta = up @ down * alpha
                        delta = (up_weight @ down_weight) * alpha
                        module.weight += delta.to(module.weight.device, dtype=module.weight.dtype)
            else:
                print(f"[Warning] Module not found: {prefix}")
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
lora_path           = '/home/lgk/2024/2031/EasyAnimate-8_19/models/checkpoint-2000.safetensors'
# Other params
sample_size     = [512, 512]
weight_dtype    = torch.bfloat16
prompt          = "Anode Carbon Block Surface Residue Image"
#prompt          = "1man, bangs, blue eyes, blunt bangs, blurry, blurry background, bob cut, depth of field, lips, looking at viewer, motion blur, nose, realistic, red lips, shirt, short hair, solo, white shirt."
negative_prompt = "bad detailed"
guidance_scale  = 6.0
seed            = 43
num_images      = 2  # 设置要生成的图像数量
lora_weight     = 0.95
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
'''
if lora_path is not None:
    #pipeline = merge_lora(pipeline, lora_path, lora_weight)
    pipeline = PixArtAlphaPipeline.from_pretrained(
        "models/Diffusion_Transformer/PixArt-XL-2-512x512",
        torch_dtype=torch.bfloat16
    ).to("cuda")

    load_lora_into_transformer(pipeline.transformer, "models/checkpoint-2000.safetensors", alpha=1.0)

# 为每张图生成不同的种子（可选：也可固定种子以复现）
generators = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(num_images)]
'''
# ... 前面的 transformer, vae, scheduler, pipeline 构建代码保持不变 ...

pipeline.to("cuda")
pipeline.enable_model_cpu_offload()

if lora_path is not None:
    # ✅ 修复：直接使用官方提供的 merge_lora 函数，并传入正确的变量
    print(f"正在加载 LoRA: {lora_path}，权重: {lora_weight}")
    pipeline = merge_lora(pipeline, lora_path, lora_weight)
    
    # 删除下面那些重新加载 pipeline 和自定义注入的错误代码：
    # pipeline = PixArtAlphaPipeline.from_pretrained(...) 
    # load_lora_into_transformer(...)

# 为每张图生成不同的种子
generators = [torch.Generator(device="cuda").manual_seed(seed + i) for i in range(num_images)]

# ... 后面的生成和保存代码保持不变 ...
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
