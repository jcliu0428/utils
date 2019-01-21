import torch

def bbox_overlaps(bboxes1,bboxes2,mode='iou',is_aligned=False):
    """Intersection over union,Intersection over foreground."""
    assert mode in ['iou','iof']
    
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows  == cols
    if rows * cols == 0:
        return bboxes1.new(rows,1) if is_aligned else bboxes1.new(rows,cols)
    
    if is_aligned:
        lt = torch.max(bboxes1[:,:2],bboxes2[:,:2])
        rb = torch.min(bboxes1[:,2:],bboxes2[:,2:])
        
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:,0] * wh[:,1]
        area1 = (bboxes1[:,2] - bboxes1[:,0] + 1) * (bboxes2[:,3] - bboxes[:,1] + 1)
        
        if  mode == 'iou':
            area2 = (bboxes2[:,2] - bboxes2[:,0] + 1) * (bboxes2[:,3] - bboxes[:,1] + 1)
            ious = overlap / (area1 + area2 - overlap)
            
        else:
            ious = overlap / area1
            
    else:
        lt = torch.max(bboxes1[:,None,:2],bboxes2[:,:2]) # [rows,cols,2]
        rb = torch.min(bboxes1[:,None,2:],bboxes2[:,2:]) # [rows,cols,2]
        
        wh = (rb - lt + 1).clamp(min=0)
        overlap = wh[:,:,0] * wh[:,:,1] # [rows,cols]
        area1 = (bboxes1[:,2] - bboxes1[:,0] + 1) * (bboxes1[:,3] - bboxes1[:,1] + 1)
        
        if mode == 'iou':
            area2 = (bboxes2[:,2] - bboxes2[:,0] + 1) * (bboxes2[:,3] - bboxes[:,1] + 1)
            ious =  overlap / (area1[:,None] + area2 - overlap)
            
        else:
            ious = overlap / (area1[:,None])
            
    return ious
