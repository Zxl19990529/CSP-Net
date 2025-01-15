import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler,DataLoader
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from models import deeplabV3p,deeplabV2,deeplabV2_r38, deeplabV3p_lorm,deeplabV2_lorm,UNet
import dataset
import transform as transform
import transform_exten

def get_model_specificcls(model_type,distributed,args,local_rank,device,checkpoint=None):
    model = None
    if model_type == 'deeplabv3p':
        model = deeplabV3p.Res_Deeplab(args.modelclasses,args.layers)
    elif model_type == 'deeplabv2':
        model = deeplabV2.Res_Deeplab(args.modelclasses,args.layers)
    elif model_type == 'deeplabv3p_lorm':
        model = deeplabV3p_lorm.Res_Deeplab(args.modelclasses,args.layers)
    elif model_type == 'deeplabv2_lorm':
        model = deeplabV2_lorm.Res_Deeplab(args.modelclasses,args.layers)
    elif model_type == 'deeplabv2_r38':
        model = deeplabV2_r38.Res_Deeplab(args.modelclasses)
    
    # if args.compile:
    #     model = torch.compile(model)
    if checkpoint !=None:
        if checkpoint.get('compile'):
            model = torch.compile(model)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])
    
    return model

def get_model(model_type,distributed,args,local_rank,device,checkpoint=None):
    model = None
    if model_type == 'deeplabv3p':
        model = deeplabV3p.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2':
        model = deeplabV2.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv3p_lorm':
        model = deeplabV3p_lorm.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2_lorm':
        model = deeplabV2_lorm.Res_Deeplab(args.numclasses,args.layers)
    elif model_type == 'deeplabv2_r38':
        model = deeplabV2_r38.Res_Deeplab(args.numclasses)
    elif model_type == 'UNet':
        model = UNet.UNet(n_channels=3,n_classes=args.numclasses)

    
    # if args.compile:
    #     model = torch.compile(model)
    if checkpoint !=None:
        if checkpoint.get('compile'):
            model = torch.compile(model)
        model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(model,device_ids=[local_rank])
    
    return model


def get_dataloader(args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform_exten.Compose([
            transform_exten.RandScale([0.5, 2.0]),
            transform_exten.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform_exten.RandomGaussianBlur(),
            transform_exten.RandomHorizontalFlip(),
            transform_exten.Crop([args.crop_size, args.crop_size], crop_type='rand', padding=mean, ignore_label=255),
            transform_exten.ToTensor(),
            transform_exten.Normalize(mean=mean, std=std)])
    train_transform_base = transform.Compose([
            transform.RandScale([0.5, 2.0]),
            transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([args.crop_size, args.crop_size], crop_type='rand', padding=mean, ignore_label=255),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_transform = transform.Compose([
        transform.Crop([args.crop_size, args.crop_size], crop_type='center', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    if args.dataset == 'ScribbleOnly':
        train_dataset = dataset.SingleLabelData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform_base, path = args.train_path)    
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'PseudoOnly':
        train_dataset = dataset.SingleLabelData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform_base, path = args.pesudo_path)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'ScribblePseudo':
        train_dataset = dataset.ScribblePseudoData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'ScribblePseudoDs':
        train_dataset = dataset.ScribblePseudoDsData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path,path_distancemaps=args.distancemap_s)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'ScribblePseudoDc':
        train_dataset = dataset.ScribblePseudoDcData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path,path_distancemapc=args.distancemap_c)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'ScribblePseudoDsDc':
        train_dataset = dataset.ScribblePseudoDsDcData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path, path_pesudo=args.pesudo_path,path_distancemaps=args.distancemap_s,path_distancemapc=args.distancemap_c)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'Scribble3Scribble':
        train_dataset = dataset.ScribblePseudoDsData(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path = args.train_path1, path_pesudo=args.train_path2,path_distancemaps=args.train_path3)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')
    elif args.dataset == 'Scribble3RandomScribble':
        train_dataset = dataset.Scribble3RandomScribble(split='train', data_root=args.dataset_path, data_list='train.txt', transform=train_transform, r1_path = args.train_path1, r2_path=args.train_path2,r3_path=args.train_path3)
        val_dataset = dataset.SingleLabelData(split='val', data_root=args.dataset_path, data_list='val.txt', transform=val_transform, path = 'SegmentationClassAug')

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_train, pin_memory=True,prefetch_factor=2,drop_last=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batchsize,sampler=sampler_val, pin_memory=True,prefetch_factor=2)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True,drop_last=True)
        val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=int(args.batchsize), shuffle=False, pin_memory=True)
    
    return train_loader,val_loader

