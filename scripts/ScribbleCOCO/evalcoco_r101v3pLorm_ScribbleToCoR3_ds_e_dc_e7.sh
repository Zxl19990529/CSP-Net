CUDA_VISIBLE_DEVICES=0 python ./eval_coco.py \
--layers 101 \
--batchsize 14 \
--dataset_path dataset/ScribbleCOCO2014 \
--numclasses 81 \
--workers 4 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ScribbleCOCO/train_r101v3pLorm_ScribbleToCoR3_ds_e_dc_e7/Jun21_05-48-05_ubuntu/last_checkpoint.pth \
--save_path ./log_ScribbleCOCO/evalcoco_r101v3pLorm_ScribbleToCoR3_ds_e_dc_e7