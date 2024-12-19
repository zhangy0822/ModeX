python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path lavis/projects/blip/eval/nlvr_eval.yaml
# python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip/eval/nlvr_eval.yaml
