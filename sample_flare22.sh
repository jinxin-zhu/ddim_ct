# !/bin/bash -e

cd /home/jinxinzhu/project/model_inf/ddim
# CUDA_VISIBLE_DEVICES=0
# nohup \
# python \
# -m torch.distributed.run \
# --nproc_per_node=1 \
# --master_port 45633 \
# main.py \
#     --config /home/jinxinzhu/project/model_inf/ddim/configs/flare22.yml \
#     --exp /home/jinxinzhu/project/model_inf/ddim/output \
#     --doc output1 \
#     --ni \
# > ./output/trainlog/ddim3D.log 2>&1 &
    
# nohup \
python main.py \
    --config /home/jinxinzhu/project/model_inf/ddim/configs/flare22.yml \
    --exp /home/jinxinzhu/project/model_inf/ddim/output \
    --doc sample_resize_128_128_64 \
    --sample \
    --fid \
    --ni \
    --timesteps 500
# > ./output/trainlog/ddim3D_sample.log 2>&1 &
# python run.py --local_rank=0 > diffusion_pretrain.log 2>&1 &

# nohup torchrun main.py --config ./configs/flare22.yml --exp ./ddim/output --doc output1 --ni > ddim3D.log 2>&1 &
