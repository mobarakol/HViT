# HViT
Training command:
```
 CUDA_VISIBLE_DEVICES=1 python3.6 train.py --name surgical_scene_vit --model_type R50-ViT-B_16 --pretrained_dir R50+ViT-B_16.npz --fp16 --fp16_opt_level O2 --train_batch_size 64 --num_steps 50000 --eval_batch_size 128
```
Validation Command:
```
CUDA_VISIBLE_DEVICES=1 python3.6 valid.py --model_type R50-ViT-B_16 --pretrained_dir output/surgical_scene_vit_checkpoint.bin --eval_batch_size 200
```
