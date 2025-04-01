import os

import torch
from diffusers import FluxPipeline

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

# Memory optimizations
torch.cuda.empty_cache()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=160,
    width=160,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("flux-dev.png")
