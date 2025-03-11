import requests
import torch
from PIL import Image
from transformers import Emu3ForConditionalGeneration, Emu3Processor

# processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")
# model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", torch_dtype=torch.bfloat16, device_map="cuda")

processor = Emu3Processor.from_pretrained("BAAI/Emu3-Gen-hf")
model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Gen-hf", torch_dtype="bfloat16", device_map="auto", attn_implementation="flash_attention_2")


inputs = processor(
    text=["a portrait of young girl. masterpiece, film grained, best quality.", "a dog running under the rain"],
    padding=True,
    return_tensors="pt",
    return_for_image_generation=True,
)
inputs = inputs.to(device="cuda:1", dtype=torch.bfloat16)

neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="pt").to(device="cuda:1")

image_sizes = inputs.pop("image_sizes")
HEIGHT, WIDTH = image_sizes[0]
VISUAL_TOKENS = model.vocabulary_mapping.image_tokens

def prefix_allowed_tokens_fn(batch_id, input_ids):
    height, width = HEIGHT, WIDTH
    visual_tokens = VISUAL_TOKENS
    image_wrapper_token_id = torch.tensor([processor.tokenizer.image_wrapper_token_id], device=model.device)
    eoi_token_id = torch.tensor([processor.tokenizer.eoi_token_id], device=model.device)
    eos_token_id = torch.tensor([processor.tokenizer.eos_token_id], device=model.device)
    pad_token_id = torch.tensor([processor.tokenizer.pad_token_id], device=model.device)
    eof_token_id = torch.tensor([processor.tokenizer.eof_token_id], device=model.device)
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[0]

    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (width + 1) == 0:
        return (eol_token_id, )
    elif offset == (width + 1) * height + 1:
        return (eof_token_id, )
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id, )
    elif offset == (width + 1) * height + 3:
        return (eos_token_id, )
    elif offset > (width + 1) * height + 3:
        return (pad_token_id, )
    else:
        return visual_tokens


out = model.generate(
    **inputs,
    max_new_tokens=50_000, # make sure to have enough tokens for one image
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    negative_prompt_ids=neg_inputs.input_ids, # indicate for Classifier-Free Guidance
    negative_prompt_attention_mask=neg_inputs.attention_mask,
)

image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1]: ], height=HEIGHT, width=WIDTH)
images = processor.postprocess(list(image.float()), return_tensors="PIL.Image.Image") # internally we convert to np but it's not supported in bf16 precision
for i, image in enumerate(images['pixel_values']):
    image.save(f"result{i}.png")
