export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.run --nproc_per_node=1 --master_port 29001 train.py --cfg-path lavis/projects/blip/train/retrieval_coco_ft.yaml
