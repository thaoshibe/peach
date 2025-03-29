# CUDA_VISIBLE_DEVICES=0 python anole.py --number_of_image 3 &
# CUDA_VISIBLE_DEVICES=1 python anole.py --number_of_image 5 &
# CUDA_VISIBLE_DEVICES=2 python anole.py --number_of_image 7 &
# CUDA_VISIBLE_DEVICES=3 python anole.py --number_of_image 9 &
# CUDA_VISIBLE_DEVICES=4 python anole_ict_indomain.py --number_of_example 1 &
# CUDA_VISIBLE_DEVICES=5 python anole_ict_indomain.py --number_of_example 2 &
# CUDA_VISIBLE_DEVICES=6 python anole_ict_indomain.py --number_of_example 3 &
# CUDA_VISIBLE_DEVICES=7 python anole_ict_indomain.py --number_of_example 4

# CUDA_VISIBLE_DEVICES=0 python test.py --config config/task_prefix4.yaml --start 0 --end 10 --iteration 5 --sks_name bo &
# CUDA_VISIBLE_DEVICES=1 python test.py --config config/task_prefix4.yaml --start 10 --end 20 --iteration 5 --sks_name bo &
# CUDA_VISIBLE_DEVICES=2 python test.py --config config/task_prefix4.yaml --start 20 --end 30 --iteration 5 --sks_name bo &
# CUDA_VISIBLE_DEVICES=3 python test.py --config config/task_prefix4.yaml --start 30 --end 40 --iteration 5 --sks_name bo &
# CUDA_VISIBLE_DEVICES=7 python test.py --config config/task_2img.yaml --start 10 --end 80 --iteration 0 --sks_name bo &
# CUDA_VISIBLE_DEVICES=4 python test.py --config config/task_2img.yaml --start 10 --end 80 --iteration 5 --sks_name bo &
# CUDA_VISIBLE_DEVICES=5 python test.py --config config/task_2img.yaml --start 10 --end 80 --iteration 10 --sks_name bo &
# CUDA_VISIBLE_DEVICES=6 python test.py --config config/task_2img.yaml --start 10 --end 80 --iteration 15 --sks_name bo 
# CUDA_VISIBLE_DEVICES=0 python test.py --config config/task_prefix4.yaml --iteration 5 --sks_name bo &

CUDA_VISIBLE_DEVICES=0 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 2 --iteration "best" &
CUDA_VISIBLE_DEVICES=1 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 4 --iteration "best" &
CUDA_VISIBLE_DEVICES=2 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 6 --iteration "best" &
CUDA_VISIBLE_DEVICES=3 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 8 --iteration "best" &
CUDA_VISIBLE_DEVICES=4 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 10 --iteration "best" &
CUDA_VISIBLE_DEVICES=5 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 12 --iteration "best" &
CUDA_VISIBLE_DEVICES=6 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 14 --iteration "best" &
CUDA_VISIBLE_DEVICES=7 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "mam" --token_len 16 --iteration "best"

