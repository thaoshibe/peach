import argparse
import glob
import os
import random

import torch
import wandb
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (AutoConfig, ChameleonForConditionalGeneration,
                          ChameleonProcessor)
from transformers.image_transforms import to_pil_image

SUBJECT_NAMES = ["tasha", "dog8", "Oasis", "Nozis", "Lindsay",
                "Yana", "Ocre", "omi-babi", "sparky", "Diva-4"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_prompt", type=bool, default=True)
    parser.add_argument("--text_prompt", type=str, default="Generater a photo of a person with long, dark hair, often seen wearing stylish, comfortable outfits.")
    # parser.add_argument("--fake_folder", type=str, default=True)
    parser.add_argument("--input_folder", type=str, default='/nobackup3/thao-data/data/dogs/test')
    parser.add_argument("--save_folder", type=str, default='./generated_images/icl_outdomain')
    parser.add_argument("--number_of_example", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    model_name = "leloy/Anole-7b-v0.1-hf"
    processor = ChameleonProcessor.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.max_position_embeddings = 15000  # Change max position embeddings

    model = ChameleonForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2" #Thao: Don't know why transformers 4.46.1 doesnt support Chameleon with this option
    ).to('cuda')
    # subjectnames = SUBJECT_NAMES[args.start:args.end]
    subjectnames = os.listdir(args.input_folder)
    subjectnames = [subjectname for subjectname in subjectnames if os.path.isdir(os.path.join(args.input_folder, subjectname))]
    if args.image_prompt:
        image_folders = [os.path.join(args.input_folder, subjectname) for subjectname in subjectnames]
    else:
        caption_file = 'subject-detailed-captions.json'
        with open(caption_file, 'r') as f:
            captions = json.load(f)
    os.makedirs(args.save_folder, exist_ok=True)
    saving_index = 0
    for index, subjectname in enumerate(tqdm(subjectnames)):
        image_folder = image_folders[index]

        # full_prompt = f"This is Max {image_chunk}. Another photo of Max"
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
        # count number of image used:
        num_images_used = args.number_of_example*2 + 1
        gt_image_path = [image_paths[0]]
        ref_one_image_path = sorted(glob.glob(os.path.join(args.input_folder, 'dog8', '*.jpg')))[:2]
        ref_two_image_path = sorted(glob.glob(os.path.join(args.input_folder, 'B.Atis', '*.jpg')))[:2]
        print("ref_one_image_path", ref_one_image_path)
        print("ref_two_image_path", ref_two_image_path)
        breakpoint()
        if args.number_of_example==1:
            image_list = ref_one_image_path + gt_image_path
            image = [Image.open(image_path).convert("RGB") for image_path in image_list]
            ict_example = f"This is a photo of Luna: <image>. This is another photo of Luna: <image>\n"
        elif args.number_of_example==2:
            image_list = ref_one_image_path +ref_two_image_path + gt_image_path
            image = [Image.open(image_path).convert("RGB") for image_path in image_list]
            ict_example = f"This is a photo of Luna: <image>. This is another photo of Luna: <image>\n" + \
                f"This is a photo of Bella: <image>. This is another photo of Bella: <image>\n"
        # duplicate the number of exmples/ demonstration
        # full_prompt = ict_example*args.number_of_example
        # add the query
        full_prompt = ict_example + 'This is a photo of Max: <image>. This is another photo of Max: '

        inputs = processor(text=[full_prompt]*1, images=[image]*1, return_tensors="pt").to(model.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
        save_path = os.path.join(args.save_folder, subjectname, str(args.number_of_example))
        os.makedirs(save_path, exist_ok=True)
        # for i in tqdm(range(0, 20, 10)):
        for j in range(5):
            generate_ids = model.generate(**inputs, multimodal_generation_mode="image-only", max_new_tokens=1026, do_sample=True)
            response_ids = generate_ids[:, inputs["input_ids"].shape[-1]:]
            pixel_values = model.decode_image_tokens(response_ids[:, 1:-1])
            pixel_values = processor.postprocess_pixel_values(pixel_values)

            image = to_pil_image(pixel_values[0].detach().cpu())
            image = image.resize((300, 300))
            image.save(os.path.join(save_path, f"{j}.png"))
                # saving_index += 1
                # print('Saved image:', os.path.join("./anole", f"{saving_index}.png"))
            #     os.makedirs(save_path, exist_ok=True)
            #     image.save(f'{save_path}/{prompt_short}_{index}.png')
            #     print(f"Saved image {index} to {save_path}/{prompt_short}_{index}.png")
            #     index += 1
            # save_location = os.path.join(args.save_folder, f"{index}.png")
            # image.save(save_location)
