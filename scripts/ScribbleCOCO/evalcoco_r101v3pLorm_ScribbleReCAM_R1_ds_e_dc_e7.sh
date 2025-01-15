CUDA_VISIBLE_DEVICES=0 python ./eval_coco.py \
--layers 101 \
--batchsize 48 \
--dataset_path dataset/ScribbleCOCO2014 \
--numclasses 81 \
--workers 8 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ScribbleCOCO/train_r101v3pLorm_ScribbleReCAM_R1_ds_e_dc_e7/Aug15_04-25-39_ubuntu/last_checkpoint.pth \
--save_path ./log_ScribbleCOCO/evalcoco_r101v3pLorm_ScribbleReCAM_R1_ds_e_dc_e7