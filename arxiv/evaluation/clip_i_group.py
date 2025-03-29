import argparse
import glob
import os

import torch
from clip_image_similarity import CLIPEvaluator
from PIL import Image
from torch.nn import CosineSimilarity
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Face verification using InsightFace")
    parser.add_argument("--fake_base", type=str, default="/mnt/localssd/code/data/yochameleon-data/train/",
                        help="Path to the folder containing fake images")
    parser.add_argument("--real_base", type=str, default="/mnt/localssd/code/data/yochameleon-data/train/",
                        help="Path to the folder containing real images")
    parser.add_argument("--output_file", type=str, default="clip_similarity.json",
                        help="Path to the output JSON file")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    evaluator = CLIPEvaluator()
    identities = os.listdir(args.fake_base)
    score_lists = []
    for identity in tqdm(identities):
        # try:
        list_real_images = glob.glob(os.path.join(args.real_base, identity, '*.png')) + glob.glob(os.path.join(args.real_base, identity, '*.jpg'))
        list_fake_images = glob.glob(os.path.join(args.fake_base, identity, '*.png')) + glob.glob(os.path.join(args.fake_base, identity, '*.jpg'))
        # print('Found', len(list_real_images), 'real images and', len(list_fake_images), 'fake images')
        similarity = evaluator.compute_similarity(list_real_images, list_fake_images, average=True)
        # print('Similarity score:', similarity)
        # print('Average similarity score:', sum(similarity) / len(similarity))
        score_lists.append(sum(similarity) / len(similarity))
        # except Exception as e:
        #     print('Error:', e)
    print('Average similarity score for all identities:', sum(score_lists) / len(score_lists))
