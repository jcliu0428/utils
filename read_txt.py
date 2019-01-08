import os
import os.path as osp
import numpy as np

"""Here,we assume that the file contains a 4x4 array."""

with open('0.txt','r') as fp:
    lines = fp.readlines()
    for i,line in enumerate(lines):
        line = line.strip() #remove '\n'
        line = line.split(' ') # make line into list type
        pose_array[i,:] = map(lambda x:float(x),line)
        
