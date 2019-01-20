import cv2

def bgr2gray(img,keepdim=True):
    """keepdim(bool):If False(by default),then return the grayscale image with 2 dims,otherwise 3 dims.
    """
    # When we call cv2.COLOR_BGR2GRAY,the dimensions of img will automatically reduce from 3 to 2.
    out_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if keepdim:
        # add the 2nd dimension to keep dim unchanged.
        out_dim = out_img[...,None]
    return out_img
    
def gray2bgr(img):
    """Convert a grayscale image to BGR image."""
    img = img[...,None] if img.ndim == 2 else img
    # When we call cv2.COLOR_GRAY2BGR,the dimensions of img will automatically imcrease from 2 to 3.
    out_img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    return out_img
    
    
def convert_color_factory(src,dst):
    """Return a function 'convert_color',whose parameter is just the image ndarray."""
    code = getattr(cv2,'COLOR_{}2{}'.format(src.upper(),src.upper()))
    
    def convert_color(img):
        out_img = cv2.cvtColor(img,code)
        return out_img
        
    return convert_color
    
    
bgr2rgb = convert_color_factory('bgr','rgb')

rgb2bgr = convert_color_factory('rgb','bgr')

bgr2hsv = convert_color_factory('bgr','hsv')

hsv2bgr = convert_color_factory('hsv','bgr')
        
