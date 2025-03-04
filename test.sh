CUDA_VISIBLE_DEVICES=0 python anole.py --number_of_image 3 &
CUDA_VISIBLE_DEVICES=1 python anole.py --number_of_image 5 &
CUDA_VISIBLE_DEVICES=2 python anole.py --number_of_image 7 &
CUDA_VISIBLE_DEVICES=3 python anole.py --number_of_image 9 &
CUDA_VISIBLE_DEVICES=4 python anole_ict_indomain.py --number_of_example 1 &
CUDA_VISIBLE_DEVICES=5 python anole_ict_indomain.py --number_of_example 2 &
CUDA_VISIBLE_DEVICES=6 python anole_ict_indomain.py --number_of_example 3 &
CUDA_VISIBLE_DEVICES=7 python anole_ict_indomain.py --number_of_example 4