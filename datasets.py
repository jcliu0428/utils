import os
import numpy as np
import scipy.io as sio

import torch

from PIL import Image
from torch.utils import data

num_classes = 21
ignore_label = 255
root = '/path/to/dataset/image/and/annotations'

np.random.seed(10)

def make_dataset(mode):
    assert mode in ['train','val','test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root,'imgs')
        mask_path = os.path.join(root,'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(root,'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path,it + '.png'),os.path.join(mask_path),it + '.mat')
            items.append(item)
            
    elif mode == 'val':
        img_path = os.path.join(root,'VOC2012')
        mask_path = os.path.join(root,'segmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(root,'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path,it + '.png'),os.path.join(mask_path),it + '.png')
            items.append(item)
            
    else:
        img_path = os.path.join(root,'VOCtest')
        data_list = [l.strip('\n') for l in open(os.path.join(root,'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path,it))
    return items
  
class VOC(data.dataset):
    def __init__(self,mode,joint_transform=None,sliding_crop=None,transform=None,target_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images,please check the dataset.')
            
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self,index):
        if self.mode == 'test':
            img_path,img_name = self.imgs[index]
            img = Image.open(os.path.join(img_path,img_name + '.png')).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name,img
        
        img_path,mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)
            
        if self.joint_transform is not None:
            img,mask = self.joint_transform(img,mask)
        
        if self.sliding_crop is not None:
            img_slices,mask_slices,slices_info = self.sliding_crop(img,mask)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            if self.target_transform is not None:
                mask_slices = [self.transform(e) for e in mask_slices]
            img,mask = torch.stack(img_slices,0),torch.stack(mask_slices,0)
            return img,mask,torch.LongTensor(slices_info)
          
        else:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            return img,mask
          
    def __len__(self):
        return len(self.imgs)
