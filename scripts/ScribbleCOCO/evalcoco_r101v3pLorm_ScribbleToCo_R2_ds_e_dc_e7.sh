CUDA_VISIBLE_DEVICES=0 python ./eval_coco.py \
--layers 101 \
--batchsize 24 \
--dataset_path dataset/ScribbleCOCO2014 \
--numclasses 81 \
--workers 4 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ScribbleCOCO/train_r101v3pLorm_ScribbleToCo_R2_ds_e_dc_e7/Aug20_05-45-29_ubuntu/last_checkpoint.pth \
--save_path ./log_ScribbleCOCO/evalcoco_r101v3pLorm_ScribbleToCo_R2_ds_e_dc_e7