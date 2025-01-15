#/bin/bash

CUDA_VISIBLE_DEVICES=0 python ./evaluate_acdc.py \
--layers 101 \
--dataset_path dataset/ramdisk/ScribbleACDC \
--dataset ScribbleOnly \
--numclasses 4 \
--workers 2 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ACDC/train_r101v3pLorm_scribbleToCo_dse_dce6/Oct29_00-44-05_GPC-10/last_checkpoint.pth \
--save_path ./log_ACDC/ScribblePseudoLormDsDc/emlc_r101v3pLorm_scribbleToCo_dse_dce6