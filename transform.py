import numpy as np
import torch
from . import image_preprocessing as img_pre

__all__ = ['ImageTransform','BboxTransform','MaskTransform','Numpy2Tensor']

class ImageTransform(object):
    def __init__(self,mean=(0,0,0),std=(1,1,1),to_rgb=True,size_divisor=None):
        self.mean = np.array(mean,dtype=np.float32)
        self.std = np.array(std,dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor
        
    def __call__(self,img,flip=False,keep_ratio=True):
        if keep_ratio:
            img,scale_factor = img_pre.imrescale(img,scale,return_scale=True)
        else:
            img,w_scale,h_scale = img_pre.imresize(img,scale,return_scale=True)
            
            scale_factor = np.array([w_scale,h_scale,w_scale,h_scale],dtype=np.float32)
            
        img_shape = img.shape
        img = img_pre.imnormalize(img,self.mean,self.std,self.to_rgb)
        if flip:
            img = img_pre.imflip(img)
        if self.size_divisor is not None:
            img = img_pre.impad_to_multiple(img,self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        
        # [channel,height,width]
        img = img.transpose(2,0,1)
        return img,img_shape,pad_shape,scale_factor
    
    
    
def bbox_flip(bboxes,img_shape):
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[...,0::4] = w - bboxes[...,2::4] - 1
    flipped[...,2::4] = w - bboxes[...,0::4] - 1
    return flipped

class BboxTransform(object):
    """rescale bboxes according to image_size,flip bboxes,pad the first dimension(N) to max_num_gts."""
    def __init__(self,max_num_gts = None):
        self.max_num_gts = max_num_gts
        
    def __call__(self,bboxes,img_shape,scale_factor,flip=False):
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes,img_shape)
        gt_bboxes[:,0::2] = np.clip(gt_bboxes[:,0::2],0,img_shape[1])
        gt_bboxes[:,1::2] = np.clip(gt_bboxes[:,1::2],0,img_shape[0])
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts,4),dtype=np.float32)
            padded_bboxes[:num_gts,:] = gt_bboxes
            return padded_bboxes
        
class MaskTransform(object):
    def __call__(self,masks,pad_shape,scale_factor,flip=False):
        masks = [img_pre.imrescale(mask,scale_factor,interpolation='nearest') for mask in masks]
        if flip:
            # flipping mask is different from flipping bboxes. Flipping mask is operated inside the bbox.
            masks = [mask[:,::-1] for mask in masks]
        padded_masks = [img_pre.impad(mask,pad_shape[:2],pad_val=0) for mask in masks]
        padded_masks = np.stack(padded_masks,axis=0)
        return padded_masks
    
class Numpy2Tensor(object):
    def __init__(self):
        pass
    # args should be iterable.
    def __call__(self,*args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
