import os
import sys
import time
import shutil
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.utils import data
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
import socket
from tool import imutils
import time
import scipy.io as io
import cv2,argparse

from models import deeplabV3p,deeplabV2,deeplabV2_r38, deeplabV3p_lorm

import basic_function as func
import dataset
import transform


# os.environ['CUDA_VISIBLE_DEVICES']='2'

parser = argparse.ArgumentParser(description='PyTorch Hierachy_dif Training')
parser.add_argument('--layers', type=int, metavar='LAYERS',default= 50, help='the layer number of resnet: 18, 34, 50, 101, 152')
parser.add_argument('--dataset_path', metavar='DATASET_PATH',default='dataset/ramdisk/ScribbleCOCO2014', help='path to the dataset(multiple paths are concated by "+")')
parser.add_argument('--numclasses', type=int, metavar='NUMCLASSES', default=81, help='number of classes')
parser.add_argument('--workers', default=4, type=int, metavar='WORKERS', help='number of dataload worker')
parser.add_argument ('--shrink_factor', default=1, type=int, metavar='SHRINK',
                                   help='shrink factor of attention map, preserved as URSS.' )
parser.add_argument('--batchsize', default=12, type=int, metavar='BATCH_SIZE', help='batchsize')
# val param
parser.add_argument('--model_type',default='deeplabv3p',type=str,help='Model type selection. nonRW|RW|res50_cam|res50_labelFusion')
parser.add_argument('--checkpoint_path', metavar='CHECKPOINT_PATH', help='path to the checkpoint file',default='log_coco2014_L40/Fully/train_r50_deeplabv3p_Fully/Mar07_00-22-13_milab-PowerEdge-T640/last_checkpoint.pth')
parser.add_argument('--save_path', default='log_coco2014_L40/Fully/emlc_r50_deeplabv3p_Fully_evalcoco', metavar='SAVE_PATH', help='path to save the visualizations')
parser.add_argument('--colormap', default='ScribbleCOCO2014', help='Choose the color map  for visualization')

# parser.add_argument('--crf',default=False,type=bool,help='whether use crf for post processing.')

args = parser.parse_args()

print(args)
def ss_test(model, inputs, scale=1):
    # print("tensor...")
    b, c, h, w = inputs.shape
    scaled_inputs = F.interpolate(inputs, size=(int(h*scale), int(w*scale)), mode="bilinear", align_corners=True)
    # print(inputs.shape,scaled_inputs.shape,scale)
    outputs = model.forward_eval(scaled_inputs)
    # torch.cuda.synchronize()
    # outputs = outputs[-1]
    outputs = F.interpolate(outputs, size=(h, w), mode='bilinear', align_corners=True)
    return outputs

def ms_test(model, img):
    n, c, h, w = img.size()
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    # scales = [2.0]
    inputs=img
    # full_probs = torch.cuda.FloatTensor(n, 81, h, w).fill_(0)
    # full_probs = torch.zeros((n,81,h,w),dtype=torch.FloatTensor,device='cuda')
    full_probs = None
    for scale in scales:
        # print(scale)
        probs = ss_test(model,inputs, scale)
        flip_probs = ss_test(model,inputs.flip(3), scale)
        probs = probs + flip_probs.flip(3)
        full_probs = probs if full_probs == None else (full_probs+probs)
        # full_probs += probs
    return full_probs


def test(args, test_loader, model, classes, mean, std, base_size, crop_h, crop_w, scales): # base_size=512, crop_ are 465
    tbar = tqdm ( test_loader ,ncols=90,total=len(test_loader))
    confusion_meter = tnt.meter.ConfusionMeter( args.numclasses, normalized=False )
    confusion_meter.conf = np.zeros((args.numclasses, args.numclasses), dtype=np.int64) # I add the int64 here, COCO is too big, int32 will result in the -2,147,483,648
    model.eval()
    # mean_tensor = torch.tensor(mean).cuda()
    # std_tensor = torch.tensor(std).cuda()
    # mean = mean_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0).repeat(1,1,465,465) # [3] -> [3,465,465]
    # std = std_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(0).repeat(1,1,465,465)
    for i, (input_imgs, gts, img_paths) in enumerate(tbar): # input_imgs is not resized to 465,465
        input_imgs = input_imgs.cuda() # 1,c,h,w
        input_imgs = F.interpolate(input_imgs, size=(base_size), mode="bilinear", align_corners=True)
        # b,ori_h,ori_w = gt.shape
        #####---------new_eval------#
        with torch.no_grad():
            t1=time.time()
            predictions = ms_test(model,input_imgs) # 1,21,h,w
            # torch.cuda.synchronize()
            t2=time.time()
            # print(input_imgs.shape,'per image',(t2-t1))
        #---post processing with denseCRF---#
            # if args.crf:
            #     img_array = (input_imgs*std+mean).to(torch.int32)
            #     img_array = img_array.cpu().numpy().squeeze(0) # 1,3,465,465 -> 3,465,465
                
            #     img_array = np.transpose(img_array,(1,2,0)).astype(np.uint8) # 3,465,465 ->465,465,3
            #     img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)

        #---crop and resize back the predictions---#
        for k in range(len(gts)):
            img_path = img_paths[k]
            gt = gts[k].unsqueeze(0)
            b,ori_h,ori_w = gt.shape # 1,h,w
            prediction = predictions[k].unsqueeze(0)
            prediction = torch.argmax(prediction,dim=1,keepdim=True) # 1,21,512,512 -> 1,1,512,512
            prediction = F.interpolate(prediction.float(),(ori_h,ori_w),mode='nearest')
            prediction = prediction[0].long() # 1,1,512,512 -> 1,512,512

            valid_pixel = gt.ne(255)
            confusion_meter.add(prediction[valid_pixel], gt[valid_pixel])
            
            mask = func.get_mask_pallete ( prediction[0].cpu ().numpy (), args.colormap )
            img_name = img_path.split('/')[-1].split('.')[0]
            # img_name = img_path[0][img_path[0].rfind ( '/' ) + 1 :-4]
            mask.save ( os.path.join ( args.save_path, img_name + '.png' ) )
            
            # for debug the background is -1
            confusion_matrix = confusion_meter.value()
            inter = np.diag(confusion_matrix)
            union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter
            mean_iou_ind = inter/union
            if mean_iou_ind[0]<0: # dataset/ramdisk/ScribbleCOCO2014/JPEGImages/COCO_val2014_000000167155.jpg
                print(mean_iou_ind)
                print(confusion_matrix)

    confusion_matrix = confusion_meter.value()
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(1).clip(min=1e-12) + confusion_matrix.sum(0).clip(min=1e-12) - inter

    mean_iou_ind = inter/union
    mean_iou_all = mean_iou_ind.mean()
    mean_acc_pix = float(inter.sum())/float(confusion_matrix.sum())
    print(' * IOU_All {iou}'.format(iou=mean_iou_all))
    print(' * IOU_Ind {iou}'.format(iou=mean_iou_ind))
    print(' * ACC_Pix {acc}'.format(acc=mean_acc_pix))
    with open(args.save_path+'.txt','a') as f:
        f.writelines(' * IOU_All {iou}\n'.format(iou=mean_iou_all))
        f.writelines(' * IOU_Ind {iou}\n'.format(iou=mean_iou_ind))
        f.writelines(' * ACC_Pix {acc}\n'.format(acc=mean_acc_pix))
        f.close()

def collate_fn(list_items):
    img = []
    label = []
    imgpath=[]
    for img_, label_, imgpath_ in list_items:
    #  print(f'x_={x_}, y_={y_}')
        img.append(img_[0])
        label.append(label_[0])
        imgpath.append(imgpath_)
    img=torch.stack(img)
    return img,label,imgpath

if __name__ == '__main__':

    opt_manualSeed = 3407
    print("Random Seed: ", opt_manualSeed)
    np.random.seed ( opt_manualSeed )
    random.seed ( opt_manualSeed )
    torch.manual_seed ( opt_manualSeed )
    torch.cuda.manual_seed_all ( opt_manualSeed )

    # cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    # if args.crf:
    #     args.save_path +='_crf'
    if not os.path.exists ( args.save_path ) :
        os.makedirs ( args.save_path )

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    val_transform = transform.Compose ( [transform.ResizeImage((512,512)), transform.ToTensor (),transform.Normalize(mean=mean,std=std)])
    val_dataset = dataset.SemData_val(split='val', data_root=args.dataset_path, data_list='val.txt',
                                    transform=val_transform, path='SegmentationClassAug')
    val_loader = data.DataLoader ( val_dataset, num_workers=args.workers,
                                batch_size=args.batchsize, shuffle=False, pin_memory=True,collate_fn=collate_fn)
    model_type = args.model_type
    if model_type == 'deeplabv3p':
        model = deeplabV3p.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2':
        model = deeplabV2.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv3p_lorm':
        model = deeplabV3p_lorm.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2_r38':
        model = deeplabV2_r38.deeplabv2_r38(num_classes=args.numclasses)
    model_pretrain = torch.load ( args.checkpoint_path )
    if model_pretrain.get('compile'):
        model = torch.compile(model)
    # print(model_pretrain['state_dict'].keys())
    # model = func.param_restore_all ( model, model_pretrain['state_dict'] )
    model.load_state_dict(model_pretrain['state_dict'])
    model = model.cuda()

    test(args, val_loader, model, args.numclasses, mean, std, 512, 465, 465, [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

    for item in args.__dict__.items():
        print(item)