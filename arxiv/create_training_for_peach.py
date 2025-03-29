import argparse
import glob
import json
import math
import os
from itertools import combinations, permutations

from tqdm import tqdm


def get_args():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Create In Context Learning Examples")
	parser.add_argument("--input_folder", type=str, default='/nobackup3/thao-data/data/dogs/test', help="Path to the base image")
	parser.add_argument("--save_folder", type=str, default='/nobackup3/thao-data/data/dogs/json', help="Path to the base image")
	parser.add_argument("--token_length", type=int, default=16, help="Token length")
	# parser.add_argument("--spacing", type=int, default=1, help="spacing")
	parser.add_argument("--num_of_real_images", type=int, default=-100, help="spacing")
	parser.add_argument("--name", type=str, default="dog")
	parser.add_argument("--negative_image", type=bool, default=False)
	parser.add_argument("--divide_before_positive", type=bool, default=False)
	parser.add_argument("--limit_negative", type=int, default=500)
	parser.add_argument("--consistent_prompt", type=bool, default=False)
	return parser.parse_args()

# Read the JSON file
def get_personalized_prompt(token_length=1, identifier=16200, index=16201):
	prefix_tokens = [f'<reserved{index+i}>' for i in range(token_length)]
	personalized_tokens = [f'<reserved{identifier}>']
	personalized_tokens.extend(prefix_tokens)
	if identifier == 16200:
		sks_prompt = f"{personalized_tokens[0]} is {''.join(personalized_tokens[1:])}. A photo of <reserved{identifier}>."
	else:
		sks_prompt = f"A photo of {''.join(personalized_tokens[1:])}."
	return sks_prompt

def duplicate_list_to_match_size(lst, k):
	# Repeat the list until it matches the size of k
	repeated_list = (lst * (k // len(lst) + 1))[:k]
	return repeated_list

if __name__ == "__main__":
	args = get_args()
	
	# Thao: Uncomment this if you want to use the scores.json file
	real_images = glob.glob(os.path.join(args.input_folder, "**/*.png"))
	for ext in ['jpg', 'jpeg', 'jpeg', 'JPG', 'JPEG', 'JPEG']:
		real_images.extend(glob.glob(os.path.join(args.input_folder, f"**/*.{ext}")))

	data = []
	total_data_point = 0
	dog_identities = os.listdir(args.input_folder)
	for this_dog in dog_identities:
		real_images = glob.glob(os.path.join(args.input_folder, this_dog, "*.png"))
		for ext in ['jpg', 'jpeg', 'jpeg', 'JPG', 'JPEG', 'JPEG']:
			real_images.extend(glob.glob(os.path.join(args.input_folder, this_dog, f"*.{ext}")))
		
		comb_images = list(combinations(real_images, 2))
		# comb_images = list(combinations(real_images, 4))
		# print(len(real_images), len(comb_images))
		total_data_point += len(comb_images)
		
		for image_paths in comb_images:
			conv = [
				{
				"from": "human",
				# "value": f"<image><TASK_PROMPT><image>\n<image><TASK_PROMPT>"
				"value": f"<image><TASK_PROMPT>"
				},
				{
				"from": "bot",
				"value": f"<image>"
				},
			]
			data.append({
				"image": [x for x in image_paths],
				"conversations": conv
			})
	os.makedirs(args.save_folder, exist_ok=True)
	print(f"Total data points: {total_data_point}")
	save_location = os.path.join(args.save_folder, f'{args.name}.json')
	with open(save_location, 'w') as f:
		json.dump(data, f)
	print(f"Saved conversation at: {save_location}")
