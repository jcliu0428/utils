import numpy as np

def bbox_area(boxes):
    w = (boxes[:,2]-boxes[:,0]+1)
    h = (boxes[:,3]-boxes[:,1]+1)
    area = w*h

    neg_area_idx = np.where(areas<0)[0]
    if neg_area_idx.size:
        raise ValueError('There exists negative areas bboxes.')
    return areas,neg_area_idx

def xywh_to_xyxy(boxes):
    if isinstace(boxes,(list,tuple)):
        assert len(boxes) == 4
        x1,y1=boxes[0],boxes[1]
        x2 = x1 + np.maximum(0,boxes[2]-1)
        y2 = y1 + np.maximum(0,boxes[3]-1)
        return (x1,y1,x2,y2)

    elif isinstance(boxes,np.ndarray):
        return np.hstack((boxes[:,0:2],boxes[:,0:2]+np.maximum(0,boxes[:,2:4]-1))


def xyxy_to_xywh(boxes):
    if isinstance(boxes,(list,tuple)):
        assert len(boxes) == 4
        x1,y1,x2,y2 = (boxes[i] for i in range(4))
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx,cy,w,h)

    elif isinstance(boxes,ndarray):
        return np.hstack(((boxes[:,0:2]+boxes[:,2:4])/2,np.maximum(boxes[:,2:4]-boxes[0:2]+1)))

    else:
        raise TypeError('Type Error!')

def clip_boxes_to_image(boxes,height,width):
    x1 = np.minimum(width-1,np.maximum(0,x1))
    x2 = np.minimum(width-1,np.maximum(0,x2))
    y1 = np.minimum(height-1,np.maximum(0,y1))
    y2 = np.minimum(height-1,np,maximum(0,y2))
    return x1,y1,x2,y2
