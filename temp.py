import argparse
import glob
import os
import random

import torch
import wandb
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          ChameleonForConditionalGeneration,
                          ChameleonProcessor)
from transformers.image_transforms import to_pil_image

model_config = AutoConfig.from_pretrained("leloy/Anole-7b-v0.1-hf")
breakpoint()
