import glob
import json
import os
import re

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from transformers import (ChameleonForConditionalGeneration,
                          ChameleonProcessor, ChameleonVQVAE,
                          ChameleonVQVAEConfig, Emu3Processor)
from transformers.image_transforms import to_pil_image

# END-OF-TURN token: <reserved08706>

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

### --- THAO: THIS IS QUICK ADAPTION --- NOT RELIABLE
def chameleon_to_emu3(input_string):
    def replacer(match):
        num = int(match.group(1))
        return f"<|extra_{num - 16100}|>"
    
    # Replace all occurrences of <reserved####> with <|extra_###|>
    updated_string = re.sub(r"<reserved(\d+)>", replacer, input_string)
    return updated_string

class PersonalizedDataset(Dataset):
    def __init__(
        self,
        json_file=None,
        placeholder_token="<reserved16200>",
        center_crop=False,
        repeat=10,
        tokenizer_max_length=2048, 
        processor: ChameleonProcessor = None,
        END_OF_TURN: int = 8710,
        only_positive: bool = False,
        personalized_prompt: str = None,
        task_disjoin: bool = False,
        model_id: str = 'leloy/Anole-7b-v0.1-hf',
    ):
        self.processor = processor
        self.model_id = model_id
        self.placeholder_token = placeholder_token
        self.max_length = tokenizer_max_length
        self.personalized_prompt = personalized_prompt
        self.END_OF_TURN = END_OF_TURN
        if model_id == 'Emu3-community/Emu3-Gen-hf':
            self.chat_template = "{% for message in messages %}{% if message['from'] == 'human' %}HUMAN: {{ message['value'] }} {% elif message['from'] == 'bot' %}ASSISTANT: {% generation %}{{ message['value']}}{% endgeneration %} {% endif %}{% endfor %}"
        elif model_id == 'leloy/Anole-7b-v0.1-hf':
            self.chat_template = "{% for message in messages %}{% if not (loop.first and message['from'] != 'human') %}{{ message['value'] }}{% if not loop.last %}<reserved08706>{% endif %}{% endif %}{% endfor %}"
        else:
            raise ValueError(f"Dataloader for {model_id} is not supported yet~")
        self.task_disjoin = task_disjoin
        data = []
        try:
            for file in json_file:
                print(f"Loading {file}")
                with open(file) as f:
                    info = json.load(f)
                    data.extend(info)
        except Exception as e:
            print(e)
            print('Could you please check the json file path?')
        self.data = data
        if only_positive:
            # If only train with positive images, then filter out all the negative_example in the image path
            self.data = [d for d in self.data if 'negative_example' not in d['image'][0]]
        self.flip_transform = transforms.RandomHorizontalFlip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_paths = self.data[i]['image']
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS) for image in images]
        images = [self.flip_transform(image) for image in images]
        conv = self.data[i]['conversations']

        # Manually added personalized prompt for text-only generation and image understanding
        if conv[-1]['value'] != "<image>":
            conv[0]['value'] = f'{self.personalized_prompt} {conv[0]["value"]}'
        conversations = self.processor.apply_chat_template(conv, chat_template=self.chat_template)
        # For recogtion and text response, we need to replace <sks> with <reserved16200>
        full_text = conversations.replace("<sks>", self.placeholder_token)
        if self.model_id == 'leloy/Anole-7b-v0.1-hf':
            example = self.processor(
                text=full_text,
                images=images,
                padding="max_length",
                max_length=self.max_length,
                )
            
        elif self.model_id == 'Emu3-community/Emu3-Gen-hf':
            # ---------------------------------------------------------------
            #     TODO: This is for image generation only -- Not yet support VQA
            # ---------------------------------------------------------------
            # full_text = full_text.replace(' <image> ', '')
            full_text = chameleon_to_emu3(full_text)
            example = self.processor(
                text=full_text,
                images=images,
                padding="max_length",
                max_length=5000, # This is hardfix... Thao, please fix this
                # return_for_image_generation=True,
                return_tensors='pt'
                )
            # print(full_text)
            # example['pixel_values'] = self.processor(text='<image>', images=images, padding="max_length", max_length=self.max_length, return_tensors='pt')['pixel_values']
        else:
            raise ValueError(f"Dataloader for {self.model_id} is not supported yet~")
        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        example['pixel_values'] = example['pixel_values'][0]
        example['image_sizes'] = torch.Tensor([512, 512])

        clone_inputs = example['input_ids'].clone()
        
        if self.model_id == 'leloy/Anole-7b-v0.1-hf':
            eot_indices = (clone_inputs == self.END_OF_TURN).nonzero()[:]
            # Initialize a mask with the same shape as the tensor, filled with -100 (mask out question)
            labels = torch.full(clone_inputs.shape, -100)
            for start_idx, end_idx in zip(eot_indices[0::2]+1, eot_indices[1::2]):
                cur_labels = clone_inputs[start_idx:end_idx+1]
                # ---------------------------------------------------------------
                #     TODO: This part trying to append image_tokens
                #     But I haven't figured out how to append image tokens in the dataloader
                #     So right now, the code for "replace <image> to real vq-vae tokens" are on-the-fly with training
                # ---------------------------------------------------------------

                # check if there is any image token in the current conversation
                # check = torch.nonzero(cur_labels==START_OF_IMAGE_INDEX).shape[0]
                # if check > 0:
                #     soi_index = torch.nonzero(cur_labels==START_OF_IMAGE_INDEX).item()+1
                #     eot_index = torch.nonzero(cur_labels==END_OF_IMAGE_INDEX).item()
                #     #----
                #     image_tokens = self.vqvae.get_image_tokens(pixel_values=example['pixel_values'][None])[0]
                #     breakpoint()
                #     pixel_values = self.vqvae.decode(image_tokens[None])
                #     images = self.processor.postprocess_pixel_values(pixel_values)
                #     image = to_pil_image(images[0].detach().cpu())
                #     image.save("test.png")

                #     cur_labels[soi_index:eot_index] = image_tokens
                # replace <image> to real vq-vae tokens
                labels[start_idx:end_idx+1] = cur_labels
        elif self.model_id == 'Emu3-community/Emu3-Gen-hf':
            eot_indices = (clone_inputs == self.END_OF_TURN).nonzero()[:]
            labels = torch.full(clone_inputs.shape, -100)
            # ---------------------------------------------------------------
            #     TODO: This is for single turn conversation only -- Not yet support multi-turn
            # ---------------------------------------------------------------
            labels[eot_indices+2:] = clone_inputs[eot_indices+2:]
        else:
            raise ValueError(f"Dataloader for {self.model_id} is not supported yet~")
        example['labels'] = labels
        return example

class PersonalizedDataset_SelfPrompting(Dataset):
    def __init__(
        self,
        config,
        processor: ChameleonProcessor = None,
        personalized_prompt: str = None,
    ):
        self.config = config
        self.processor = processor

        self.sks_token = self.config.special_tokens['SKS_TOKEN']
        self.personalized_prompt = personalized_prompt
        self.END_OF_TURN = self.config.special_tokens['END_OF_TURN']

        latent_token_start = self.config.special_tokens['LATENT_TOKEN_START']
        num_latent_tokens = self.config.prefix_token

        understanding_tokens = [f'<reserved{latent_token_start+num_latent_tokens+i}>' for i in range(self.config.prefix_token)]
        generation_tokens = [f'<reserved{latent_token_start+i}>' for i in range(self.config.prefix_token)]

        self.understanding_prompt = "".join(understanding_tokens)
        self.generation_prompt = "".join(generation_tokens)
        self.max_length = self.config.tokenizer_max_length

        data = []
        try:
            for file in self.config.json_file:
                print(f"        Loading {file}")
                with open(file) as f:
                    info = json.load(f)
                    data.extend(info)
        except Exception as e:
            print(e)
            print('\n\nCould you please check the json file path?')

        self.data = data
        self.flip_transform = transforms.RandomHorizontalFlip()
        print('The personalized prompt is: ', self.personalized_prompt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        chat_template = "{% for message in messages %}{% if not (loop.first and message['from'] != 'human') %}{{ message['value'] }}{% if not loop.last %}<reserved08706>{% endif %}{% endif %}{% endfor %}"
        image_paths = self.data[i]['image']
        image_paths = [x.replace('/mnt/localssd/code/data/yochameleon-data/', '/nobackup3/thao-data/data/mini-yochameleon-data/') for x in image_paths]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [self.flip_transform(image) for image in images]

        conv = [{"from": message["from"], "value": message["value"]} for message in self.data[i]['conversations']]
        # print(image_paths, conv)
        if conv[-1]['value'] != "<image>":
            # If the task is understanding, then add understanding prompt
            conv[0]['value'] = f'{self.personalized_prompt} {conv[0]["value"]}'
            #
            #       This will include <sks> in the answer
            #
            # conv[1]['value'] = f'<sks> is {self.understanding_prompt}. {conv[1]["value"]}'

            #
            #       This will NOT include <sks> in the answer
            #
            conv[1]['value'] = f'{self.understanding_prompt}{conv[1]["value"]}'
        else:
            caption = conv[0]['value'].split('.')[0]
            conv[0]['value'] = f'{caption}{self.understanding_prompt}. A photo of {self.sks_token}.'
            #
            #       This will include <sks> in the answer
            #
            # conv[1]['value'] = f'{caption}<image>'

            #
            #       This will NOT include <sks> in the answer
            #
            caption = caption.replace(f'{self.sks_token} is ', '')
            

        conversations = self.processor.apply_chat_template(conv, chat_template=chat_template)
        full_text = conversations.replace("<sks>", self.sks_token)

        example = self.processor(
            text=full_text,
            images=images,
            padding="max_length",
            max_length=self.max_length,
            )

        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        example['pixel_values'] = example['pixel_values'][0]

        clone_inputs = example['input_ids'].clone()
        eot_indices = (clone_inputs == self.END_OF_TURN).nonzero()[:]
        
        # Initialize a mask with the same shape as the tensor, filled with -100 (mask out question)
        labels = torch.full(clone_inputs.shape, -100)
        for start_idx, end_idx in zip(eot_indices[0::2]+1, eot_indices[1::2]):
            cur_labels = clone_inputs[start_idx:end_idx+1]
            labels[start_idx:end_idx+1] = cur_labels
        example['labels'] = labels
        return example

class RecognitionData(Dataset):
    def __init__(
        self,
        sks_name,
        placeholder_token="<reserved16200>",
        image_folder=None,
        tokenizer_max_length=1500, 
        processor: ChameleonProcessor = None,
        only_positive: bool = False,
        personalized_prompt: str = None,
    ):
        self.processor = processor
        self.sks_name = sks_name
        self.placeholder_token = placeholder_token
        self.max_length = tokenizer_max_length
        self.personalized_prompt = personalized_prompt
        self.image_paths = glob.glob(os.path.join(image_folder, "*/*.png"))
        self.image_paths = [x for x in self.image_paths if 'negative_example' not in x]
        
        print(f'Found {len(self.image_paths)} images in {image_folder}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        #-- This might be useful, as later can be trained for multiple images (interleaved data)
        # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        # print(f'Loading {image_path}')
        images = [Image.open(image_path).convert("RGB")]
        question = f'{self.personalized_prompt} Can you see {self.placeholder_token} in this photo?<image><reserved08706>'
        example = self.processor(
            text=question,
            images=images,
            # padding="max_length",
            # max_length=self.max_length,
            )
        example['inputs'] = {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'pixel_values': example['pixel_values'],
        }

        if f'/{self.sks_name}/' in image_path:
            example['labels'] = ['Yes']
        else:
            example['labels'] = ['No']
        return example

class RecognitionData_SelfPrompting(Dataset):
    def __init__(
        self,
        sks_name,
        placeholder_token="<reserved16200>",
        image_folder=None,
        tokenizer_max_length=1500, 
        processor: ChameleonProcessor = None,
        only_positive: bool = False,
        personalized_prompt: str = None,
        understanding_prompt: str = None,
    ):
        self.processor = processor
        self.sks_name = sks_name
        self.placeholder_token = placeholder_token
        self.max_length = tokenizer_max_length
        self.personalized_prompt = personalized_prompt
        self.image_paths = glob.glob(os.path.join(image_folder, "*/*.png"))
        self.image_paths = [x for x in self.image_paths if 'negative_example' not in x]
        self.understanding_prompt = understanding_prompt
        print(f'Found {len(self.image_paths)} images in {image_folder}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path = self.image_paths[i]
        #-- This might be useful, as later can be trained for multiple images (interleaved data)
        # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        # print(f'Loading {image_path}')
        images = [Image.open(image_path).convert("RGB")]
        question = f'{self.personalized_prompt} Is {self.placeholder_token} in this photo?<image>'

        example = self.processor(
            text=question,
            images=images,
            # padding="max_length",
            # max_length=self.max_length,
            )
        example['inputs'] = {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'pixel_values': example['pixel_values'],
        }

        if f'/{self.sks_name}/' in image_path:
            example['labels'] = ['Yes']
        else:
            example['labels'] = ['No']
        return example

class PersonalizedDataset_DOGS(Dataset):
    def __init__(
        self,
        config,
        processor: ChameleonProcessor = None,
        personalized_prompt: str = None,
    ):
        self.config = config
        self.processor = processor

        self.sks_token = self.config.special_tokens['SKS_TOKEN']
        self.personalized_prompt = personalized_prompt
        self.END_OF_TURN = self.config.special_tokens['END_OF_TURN']

        latent_token_start = self.config.special_tokens['LATENT_TOKEN_START']
        num_latent_tokens = self.config.prefix_token

        understanding_tokens = [f'<reserved{latent_token_start+num_latent_tokens+i}>' for i in range(self.config.prefix_token)]
        generation_tokens = [f'<reserved{latent_token_start+i}>' for i in range(self.config.prefix_token)]

        self.understanding_prompt = "".join(understanding_tokens)
        self.generation_prompt = "".join(generation_tokens)
        self.max_length = self.config.tokenizer_max_length
        self.task_prompt = self.understanding_prompt

        data = []
        try:
            for file in self.config.json_file:
                print(f"        Loading {file}")
                with open(file) as f:
                    info = json.load(f)
                    data.extend(info)
        except Exception as e:
            print(e)
            print('\n\nCould you please check the json file path?')

        self.data = data
        self.flip_transform = transforms.RandomHorizontalFlip()
        print('The personalized prompt is: ', self.personalized_prompt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        chat_template = "{% for message in messages %}{% if not (loop.first and message['from'] != 'human') %}{{ message['value'] }}{% if not loop.last %}<reserved08706>{% endif %}{% endif %}{% endfor %}"
        image_paths = self.data[i]['image']
        # image_paths = [x.replace('/mnt/localssd/code/data/yochameleon-data/', '/nobackup3/thao-data/data/mini-yochameleon-data/') for x in image_paths]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images = [self.flip_transform(image) for image in images]

        conv = [{"from": message["from"], "value": message["value"]} for message in self.data[i]['conversations']]

        caption = conv[0]['value'].split('.')[0]

        ################################################
        #
        #         <image>TASK_PROMPT<image>
        #
        ################################################
        # breakpoint()
        conv[0]['value'] = caption.replace('<TASK_PROMPT>', self.task_prompt)

        #
        #       This will include <sks> in the answer
        #
        # conv[1]['value'] = f'{caption}<image>'

        #
        #       This will NOT include <sks> in the answer
        #
        # caption = caption.replace(f'{self.sks_token} is ', '')
            

        conversations = self.processor.apply_chat_template(conv, chat_template=chat_template)
        # full_text = conversations.replace("<sks>", self.sks_token)
        # print(conversations)
        example = self.processor(
            text=conversations,
            images=images,
            padding="max_length",
            max_length=self.max_length,
            )
        example['input_ids'] = example['input_ids'][0]
        example['attention_mask'] = example['attention_mask'][0]
        # example['pixel_values'] = example['pixel_values']

        clone_inputs = example['input_ids'].clone()
        eot_indices = (clone_inputs == self.END_OF_TURN).nonzero()[:]
        
        # Initialize a mask with the same shape as the tensor, filled with -100 (mask out question)
        labels = torch.full(clone_inputs.shape, -100)
        for start_idx, end_idx in zip(eot_indices[0::2]+1, eot_indices[1::2]):
            cur_labels = clone_inputs[start_idx:end_idx+1]
            labels[start_idx:end_idx+1] = cur_labels
        example['labels'] = labels
        return example

if __name__ == "__main__":
    #-- This is for debugging purpose
    config_file = './config/task_prefix4.yaml'
    config_dict = yaml.safe_load(open(config_file, 'r'))
    config = Config(config_dict)
    config.sks_name = 'bo'
    if config.model_id == 'leloy/Anole-7b-v0.1-hf':
        processor = ChameleonProcessor.from_pretrained(config.model_id)
    elif config.model_id == 'Emu3-community/Emu3-Gen-hf':
        processor = Emu3Processor.from_pretrained(config.model_id)

    config.json_file = [x.replace('SKS_NAME', config.sks_name) for x in config.json_file]
    config.sks_name = 'bo'
    prefix_tokens = [f'<reserved{16201+i}>' for i in range(config.prefix_token)]
    personalized_tokens = [f'<reserved16200>'] + prefix_tokens
    personalized_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}."

    if config.task_disjoin:
        train_dataset = PersonalizedDataset_Disjoin(
            json_file=config.json_file,
            processor=processor,
            tokenizer_max_length=config.tokenizer_max_length,
            END_OF_TURN=config.special_tokens["END_OF_TURN"],
            personalized_prompt=personalized_prompt,
            task_disjoin=config.task_disjoin,
            )
    elif config.task_prompt:
        train_dataset = PersonalizedDataset_DOGS(
            config,
            processor=processor,
            )
    elif config.self_prompting:
        train_dataset = PersonalizedDataset_SelfPrompting(
            config,
            processor=processor,
            )
    else:
        train_dataset = PersonalizedDataset(
                json_file=config.json_file,
                processor=processor,
                tokenizer_max_length=config.tokenizer_max_length,
                END_OF_TURN=config.special_tokens["END_OF_TURN"],
                personalized_prompt=personalized_prompt,
                task_disjoin=config.task_disjoin,
                model_id=config.model_id
                )
    for i in range(len(train_dataset)):
        # breakpoint()
        print(train_dataset.__getitem__(i)['labels'].shape, train_dataset.__getitem__(i)['pixel_values'].shape)
        # train_dataset.__getitem__(i)
