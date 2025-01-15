export CUDA_DEVICE_ORDER="PCI_BUS_ID"
CUDA_VISIBLE_DEVICES=0 python ./eval_coco.py \
--layers 101 \
--batchsize 24 \
--dataset_path dataset/ScribbleCOCO2014 \
--numclasses 81 \
--workers 8 \
--model_type deeplabv3p_lorm \
--shrink_factor 1 \
--checkpoint_path ./log_ScribbleCOCO/train_r101v3pLorm_Scribblecutmix_R2_ds_e_dc_e7/Aug18_13-58-46_ubuntu/last_checkpoint.pth \
--save_path ./log_ScribbleCOCO/evalcoco_r101v3pLorm_Scribblecutmix_R2_ds_e_dc_e7