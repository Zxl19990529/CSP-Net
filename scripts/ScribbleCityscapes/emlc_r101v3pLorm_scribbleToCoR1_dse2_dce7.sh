#/bin/bash

# baseline no mask
CUDA_VISIBLE_DEVICES=0 python ./evaluate_cityscapes.py \
--layers 101 \
--dataset_path dataset/ramdisk/ScribbleCityscapes \
--dataset ScribblePseudoDsDc \
--numclasses 9 \
--workers 2 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path log_ScribbleCityscapes/train_r101v3pLorm_scribbleToCoR1_dse2_dce7/Oct21_02-57-21_GPC-10/last_checkpoint.pth \
--save_path ./log_ScribbleCityscapes/emlc_r101v3pLorm_scribbleToCoR1_dse2_dce7
