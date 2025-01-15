import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def load_image_label_list_from_npy(img_name_list, data_root):
    cls_labels_dict = np.load('{}/cls_labels.npy'.format(data_root), allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]

class ScribblePseudoDsDcData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset',path_distancemaps='distancemap_s_lambd_1',path_distancemapc='distancemap_c_lambd_e'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.pseudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.ds_names = ['{}/{}/{}.png'.format(data_root, path_distancemaps, i) for i in self.indices]
        self.dc_names = ['{}/{}/{}.png'.format(data_root, path_distancemapc, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index] 
        label_pesudo_path = self.pseudo_names[index]
        ds_path = self.ds_names[index] # distance map scribble path
        dc_path = self.dc_names[index]# distance map cam path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        distancemap_s = cv2.imread(ds_path,cv2.IMREAD_GRAYSCALE)
        distancemap_c = cv2.imread(dc_path,cv2.IMREAD_GRAYSCALE)
        # print(dc_path)
        # print(type(image),type(label),type(label_pesudo),type(distancemap_s),type(distancemap_c))
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1] or image.shape[1] != distancemap_s.shape[1] or image.shape[1] != distancemap_c.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo,distancemap_s,distancemap_c = self.transform(image, label, label_pesudo,distancemap_s,distancemap_c )
            distancemap_s = distancemap_s.float()
            distancemap_c = distancemap_c.float()
        return image, label,label_pesudo,distancemap_s,distancemap_c, image_path

class ScribblePseudoDsData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset',path_distancemaps='distancemap_s_lambd_1'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.pseudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.ds_names = ['{}/{}/{}.png'.format(data_root, path_distancemaps, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        label_pesudo_path = self.pseudo_names[index]
        ds_path = self.ds_names[index] # distance map scribble path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        distancemap_s = cv2.imread(ds_path,cv2.IMREAD_GRAYSCALE)
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1] or image.shape[1] != distancemap_s.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo,distancemap_s,_ = self.transform(image, label, label_pesudo,distancemap_s,distancemap_s )
            distancemap_s = distancemap_s.float()
        return image, label,label_pesudo,distancemap_s, image_path

class ScribblePseudoDcData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset',path_distancemapc='distancemap_c_lambd_1'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.pseudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.dc_names = ['{}/{}/{}.png'.format(data_root, path_distancemapc, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        label_pesudo_path = self.pseudo_names[index]
        dc_path = self.dc_names[index] # distance map scribble path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        distancemap_c = cv2.imread(dc_path,cv2.IMREAD_GRAYSCALE)
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1] or image.shape[1] != distancemap_c.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo,distancemap_c,_ = self.transform(image, label, label_pesudo,distancemap_c,distancemap_c )
            distancemap_c = distancemap_c.float()
        return image, label,label_pesudo,distancemap_c, image_path
    
class ScribblePseudoData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'pascal_2012_scribble',path_pesudo = 'bmp_ws_train_aug_dataset'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.pseudo_names = ['{}/{}/{}.png'.format(data_root, path_pesudo, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        label_pesudo_path = self.pseudo_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        label_pesudo = cv2.imread(label_pesudo_path,cv2.IMREAD_GRAYSCALE)
        
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1] or image.shape[1] != label_pesudo.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label,label_pesudo,_,_ = self.transform(image, label, label_pesudo,label_pesudo,label_pesudo )
        return image, label,label_pesudo, image_path

class SingleLabelData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform # see transform.py

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label, image_path
    
class SemData_boarder(Dataset):
    # this is used for the multiscale test, resize the original image to the same size as the one in training stage, which is 465*465, maintaining the original width-height ratio. The label keeps the original shape, so the predictions should be resized back to the original shape for evaluation.
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug',mean=[0,0,0],std=[0,0,0],size=[465,465]):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform
        self.mean = mean
        self.std = std
        self.target_size = size # [465,465] h,w
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        ori_h, ori_w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        
        if ori_h > ori_w:
            new_h = self.target_size[0]
            new_w = round(new_h * (ori_w/ori_h))
            pad_w = self.target_size[1] - new_w
            pad_w_half = int(pad_w/2)
            image = cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_LINEAR)            
            image = cv2.copyMakeBorder(image,0,0,pad_w_half,pad_w - pad_w_half,cv2.BORDER_CONSTANT, value=self.mean)
        elif ori_h < ori_w:
            new_w = self.target_size[1]
            new_h = round(new_w * (ori_h/ori_w))
            pad_h = self.target_size[0] - new_h
            pad_h_half = int(pad_h/2)
            image = cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_LINEAR)            
            image = cv2.copyMakeBorder(image,pad_h_half,pad_h - pad_h_half,0,0,cv2.BORDER_CONSTANT, value=self.mean)
        elif ori_h == ori_w:
            padding = self.target_size[0] - ori_h
            image = cv2.resize(image,self.target_size,interpolation=cv2.INTER_LINEAR)            
            # image = cv2.copyMakeBorder(image,padding,padding,padding,padding,cv2.BORDER_CONSTANT, value=self.mean)
        if self.transform is not None:
            if self.split == 'train':
                image, label,_,_,_ = self.transform(image, label,label,label,label)
            elif self.split == 'val':
                image, label = self.transform(image, label)
                

        return image, label, image_path
    
class SemData_val(Dataset):
    # this is used for the multiscale test, resize the original image to the same size as the one in training stage, which is 465*465, maintaining the original width-height ratio. The label keeps the original shape, so the predictions should be resized back to the original shape for evaluation.
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , path = 'SegmentationClassAug'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.lab_names = ['{}/{}/{}.png'.format(data_root, path, i) for i in self.indices]
        self.transform = transform
        # self.target_size = size # [465,465] h,w
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        label_path=self.lab_names[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        ori_h, ori_w, c = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        
        if self.transform is not None:
            if self.split == 'train':
                image, label,_,_,_ = self.transform(image, label,label,label,label)
            elif self.split == 'val':
                image, label = self.transform(image, label)
                

        return [image], [label], image_path
    

# Scribble3RandomScribble
class Scribble3RandomScribble(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None , r1_path = 'coco2014_train_scribble_r1',r2_path = 'coco2014_train_scribble_r2',r3_path='coco2014_train_scribble_r3'):
        self.split = split
        self.indices = open('{}/ImageSets/SegmentationAug/{}'.format(data_root, data_list),'r').read().splitlines()
        self.img_names = ['{}/JPEGImages/{}.jpg'.format(data_root, i) for i in self.indices]
        self.r1_names = ['{}/{}/{}.png'.format(data_root, r1_path, i) for i in self.indices]
        self.r2_names = ['{}/{}/{}.png'.format(data_root, r2_path, i) for i in self.indices]
        self.r3_names = ['{}/{}/{}.png'.format(data_root, r3_path, i) for i in self.indices]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #image_path, label_path = self.data_list[index]
        image_path=self.img_names[index]
        r1_path=self.r1_names[index]
        r2_path = self.r2_names[index]
        r3_path = self.r3_names[index] # distance map scribble path
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # image = np.float32(image)
        r1_img = cv2.imread(r1_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        r2_img = cv2.imread(r2_path,cv2.IMREAD_GRAYSCALE)
        r3_img = cv2.imread(r3_path,cv2.IMREAD_GRAYSCALE)
        
        if image.shape[0] != r1_img.shape[0] or image.shape[1] != r1_img.shape[1] or image.shape[1] != r2_img.shape[1] or image.shape[1] != r3_img.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + r1_path + "\n"))
        if self.transform is not None:
            image, r1_img,r2_img,r3_img,_ = self.transform(image, r1_img,r2_img,r3_img,r3_img )
        random_choice_list = [r1_img,r2_img,r3_img]
        selected_scribble = np.random.choice(random_choice_list,1,p=[0.3333333,0.3333333,0.3333333])

        return image, selected_scribble, image_path