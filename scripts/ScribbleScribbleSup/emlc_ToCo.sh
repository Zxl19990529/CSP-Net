#/bin/bash

# baseline no mask
CUDA_VISIBLE_DEVICES=0 python ./evaluate.py \
--layers 101 \
--dataset_path dataset/ScribbleSup/VOC2012 \
--numclasses 21 \
--workers 4 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ScribbleSup/train_r101v3pLorm_ScribbleToCo_ds_e_dc_e7/Jun07_13-40-07_ubuntu/last_checkpoint.pth \
--save_path ./log_ScribbleSup/emlc_r101v3pLorm_ScribbleToCo_ds_e_dc_e7