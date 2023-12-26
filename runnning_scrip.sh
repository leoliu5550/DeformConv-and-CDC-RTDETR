export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml

