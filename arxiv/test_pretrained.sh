SKS_NAME="mam"

CUDA_VISIBLE_DEVICES=0 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
CUDA_VISIBLE_DEVICES=1 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
CUDA_VISIBLE_DEVICES=2 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
CUDA_VISIBLE_DEVICES=3 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
CUDA_VISIBLE_DEVICES=4 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
CUDA_VISIBLE_DEVICES=5 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
CUDA_VISIBLE_DEVICES=6 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
CUDA_VISIBLE_DEVICES=7 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

# SKS_NAME="mam"

# CUDA_VISIBLE_DEVICES=0 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
# CUDA_VISIBLE_DEVICES=1 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
# CUDA_VISIBLE_DEVICES=2 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
# CUDA_VISIBLE_DEVICES=3 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
# CUDA_VISIBLE_DEVICES=4 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
# CUDA_VISIBLE_DEVICES=5 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
# CUDA_VISIBLE_DEVICES=6 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
# CUDA_VISIBLE_DEVICES=7 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

# SKS_NAME="willinvietnam"

# CUDA_VISIBLE_DEVICES=0 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
# CUDA_VISIBLE_DEVICES=1 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
# CUDA_VISIBLE_DEVICES=2 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
# CUDA_VISIBLE_DEVICES=3 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
# CUDA_VISIBLE_DEVICES=4 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
# CUDA_VISIBLE_DEVICES=5 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
# CUDA_VISIBLE_DEVICES=6 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
# CUDA_VISIBLE_DEVICES=7 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

# SKS_NAME="ciin"

# CUDA_VISIBLE_DEVICES=0 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
# CUDA_VISIBLE_DEVICES=1 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
# CUDA_VISIBLE_DEVICES=2 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
# CUDA_VISIBLE_DEVICES=3 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
# CUDA_VISIBLE_DEVICES=4 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
# CUDA_VISIBLE_DEVICES=5 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
# CUDA_VISIBLE_DEVICES=6 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
# CUDA_VISIBLE_DEVICES=7 python test_yoc.py --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

#### -- Reverse tokens

SKS_NAME="ciin"

CUDA_VISIBLE_DEVICES=0 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
CUDA_VISIBLE_DEVICES=1 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
CUDA_VISIBLE_DEVICES=2 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
CUDA_VISIBLE_DEVICES=3 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
CUDA_VISIBLE_DEVICES=4 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
CUDA_VISIBLE_DEVICES=5 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
CUDA_VISIBLE_DEVICES=6 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
CUDA_VISIBLE_DEVICES=7 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

SKS_NAME="willinvietnam"

CUDA_VISIBLE_DEVICES=0 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 2 --iteration "best" &
CUDA_VISIBLE_DEVICES=1 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 4 --iteration "best" &
CUDA_VISIBLE_DEVICES=2 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 6 --iteration "best" &
CUDA_VISIBLE_DEVICES=3 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 8 --iteration "best" &
CUDA_VISIBLE_DEVICES=4 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 10 --iteration "best" &
CUDA_VISIBLE_DEVICES=5 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 12 --iteration "best" &
CUDA_VISIBLE_DEVICES=6 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 14 --iteration "best" &
CUDA_VISIBLE_DEVICES=7 python test_yoc.py --reverse True --config config/basic.yaml --exp_name 1000 --savedir ../ckpt --sks_name "$SKS_NAME" --token_len 16 --iteration "best"

