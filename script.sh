# 1 GPU
CUDA_VISIBLE_DEVICES=0 python train.py --opt_path train.yml

# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=17654 train.py --opt_path train.yml

# 3 GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11345 train.py --opt_path train.yml

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py --opt_path train.yml



