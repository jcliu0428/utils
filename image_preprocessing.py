from __future__ import division

import cv2
from .colorspace import bgr2rgb,rgb2bgr

__all__ = ['_scale_size','imresize','imsize_like','imrescale','imnormalize','imdenormalize']

def imnormalize(img,mean,std,to_rgb=True):
    img = img.astype(np.float32)
    if to_rgb:
        img = bgr2rgb(img)
    return (img-mean) / std

def imdenormalize(img,mean,std,to_bgr=True):
    img = (img*std) + mean
    if to_bgr:
        img = rgb2bgr(img)
    return img

def _scale_size(size,scale):
    """Args:size(tuple):w,h
            scale(float):Scaling factor.
       Return:tuple(int):scaled size.
    """
    w,h = size
    return int(w*float(scale)+0.5),int(h*float(scale)+0.5)
    
interp_codes = {
    'nearest':cv2.INTER_NEAREST,
    'bilinear':cv2.INTER_LINEAR,
    'bicubic':cv2.INTER_CUBIC,
    'area':cv2.INTER_AREA,
    'lanczos':cv2.INTER_LANCZOS4
}

def imresize(img,size,return_scale=False,interpolation='bilinear'):
    h,w = img.shape[:2]
    resized_img = cv2.resize(img,size,interpolation=interp_codes[interpolation])
    
    if not return_scale:
        return resized_img
        
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img,w_scale,h_scale
        
def imresize_like(img,dst_img,return_scale=False,interpolation='bilinear'):
    h,w = dst_img.shape[:2]
    return imresize(img,(w,h),return_scale,interpolation)
        
def imrescale(img,scale,return_scale=False,interpolation='bilinear'):
    """Resize the image while keeping the aspect ratio.
       Args: scale:if it is a tuple,the image will be rescaled as large as possible within the scale.
    """
    h,w = img.shape[:2]
    if isinstance(scale,(float,int)):
        if scale <= 0:
            raise ValueError('Invalid scale:{},must be positive.'.format(scale))
        scale_factor = scale
        
    elif isinstance(scale,tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge/max(h,w),
                           max_short_edge/min(h,w))
    else:
        raise ValueError('Scale must be a number or tuple of int,but got {}'.format(type(scale)))
        
    new_size = _scale_size((w,h),scale_factor)
    rescaled_img = imresize(img,new_size,interpolation = interpolation)
    if return_scale:
        return rescaled_img,scale_factor
    else:
        return rescaled_img
