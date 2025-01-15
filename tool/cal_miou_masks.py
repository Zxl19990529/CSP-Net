import numpy as np
import os
from tqdm import tqdm
import cv2
import torchnet as tnt
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gtFolder',type=str,default='dataset/ScribbleCityscapes/SegmentationClassAug')
parser.add_argument('--predFolder',type=str,default='dataset/ScribbleCityscapes/pseudolabels/ToCo_R1')
parser.add_argument('--scribbleFolder',type=str,default='dataset/ScribbleCityscapes/cityscape_scribble_r1')
parser.add_argument('--numclass',type=int,default=9)
args=parser.parse_args()

if __name__ == '__main__':
    # Example usage:
    # Ground truth masks and predicted masks are assumed to be numpy arrays of integer values indicating class labels.

    # Example ground truth masks
    log_file=open('cal_miou_masks.txt','a')
    gt_folder = args.gtFolder
    pred_folder = args.predFolder
    scribble_folder = args.scribbleFolder
    num_classes = args.numclass  # Number of classes
    flist = os.listdir(pred_folder)
    miou = []
    confusion_meter = tnt.meter.ConfusionMeter ( num_classes, normalized=False )
    for filename in tqdm(flist):
        gt_mask = torch.tensor(cv2.imread(os.path.join(gt_folder,filename))[:,:,0],dtype=torch.long)[None,:,:]
        pred_mask = torch.tensor(cv2.imread(os.path.join(pred_folder,filename))[:,:,0],dtype=torch.long)[None,:,:]
        # valid_pixel = gt_mask!=255
        valid_pixel = gt_mask.ne(255)
        confusion_meter.add(pred_mask[valid_pixel],gt_mask[valid_pixel])

    confusion_matrix = confusion_meter.value()
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

    mean_iou_ind = inter/union
    mean_iou_all = mean_iou_ind.mean()
    mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
    print('Pred Mask Path:',pred_folder,file=log_file)
    print(' * IOU_All {iou}'.format(iou=mean_iou_all),file=log_file)
    print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind),file=log_file)
    print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix),file=log_file)
    log_file.close()