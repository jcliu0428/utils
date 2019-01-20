from __future__ import division

import cv2
from .colorspace import bgr2rgb,rgb2bgr

__all__ = ['_scale_size','imresize','imsize_like','imrescale','imnormalize','imdenormalize',
          'imflip','imrotate','bbox_flip','bbox_scaling','imcrop','impad','impad_to_multiple']

def impad(img,shape,pad_val=0):
    """shape(tuple):Expected padding shape.
       pad_val:Values to be filled in padding areas.
       Return:ndarray:The padded image.
    """
    if not isinstance(pad_val,(int,float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1],)
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = np.empty(shape,dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0],:img.shape[1],...] = img
    return pad

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

    
def imflip(img,direction='horizonal'):
    assert direction in ['horizontal','vertical']
    if direction == 'horizontal':
        return np.flip(img,axis=1)
    else:
        return np.flip(img,axis=0)
    
def imrotate(img,angle,center=None,scale=1.0,border_value=0,auto_bound=False):
    """angle:positive values mean clockwise rotation."""
    if center is not None and auto_bound:
        raise ValueError('auto_bound conflicts with center!')
    h,w = img.shape[:2]
    if center is None:
        center = ((w-1) * 0.5,(h-1)*0.5)
    assert isinstance(center,tuple)
    
    matrix = cv2.getRotationMatrix2D(center,-angle,scale)
    if auto_bound:
        cos = np.abs(matrix[0,0])
        sin = np.abs(matrix[0,1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0,2] += (new_w - w) * 0.5
        matrix[1,2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img,matrix,(w,h),borderValue = border_value)
    return rotated

def bbox_clip(bboxes,img_shape):
    """Clip bboxes to fit the image shape.
    bboxes(ndarray):Shape(...,4*k),
    img_shape(tuple):(height,width) of the image.
    """
    assert bboxes.shape[-1] % 4 == 0
    clipped_bboxes = np.empty_like(bboxes,dtype=bboxes.dtype)
    clipped_bboxes[...,0::2] = np.maximum(np.minimum(bboxes[...,0::2],img_shape[1]-1),0)
    clipped_bboxes[...,1::2] = np.maximum(np.minimum(bboxes[...,1::2],img_shape[0]-1),0)
    
    return clipped_bboxes

def bbox_scaling(bboxes,scale,clip_shape=None):
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[...,2] - bboxes[...,0] + 1
        h = bboxes[...,3] - bboxes[...,1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw,-dh,dw,dh),axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes,clip_shape)
    
    else:
        return scaled_bboxes
    
def imcrop(img,bboxes,scale=1.0,pad_fill=None):
    """scale the bboxes -> clip bboxes -> crop and pad."""
    channel = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill,(int,float)):
            pad_fill = [pad_fill for _ in range(channel)]
        assert len(pad_fill) == channel
    # bboxes.ndim == 1 if there is only one box.
    _bboxes = bboxes[None,...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes,scale).astype(np.int32)
    clipped_bboxes = bbox_clip(scaled_bboxes,img.shape)
    
    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1,y1,x2,y2 = tuple(clipped_bbox[i,:])
        if pad_fill is None:
            patch = img[y1:y2+1,x1:x2+1,...]
        else:
            _x1,_y1,_x2,_y2 = tuple(scaled_bboxes[i,:])
            if channel == 2:
                patch_shape = (_y2 - _y1 + 1,_x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1,_x2 - _x1 + 1,chn)
            patch = np.array(pad_fill,dtype=img.dtype) * np.ones(patch_shape,dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h,x_start:x_start + w,...] = img[y1:y1+h,x1:x1+w,...]
        patches.append(patch)
        
    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches
