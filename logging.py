from collections import deque

class SmoothedValue(object):
    def __init__(self,window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0
        
    def AddValue(self,value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value
        
    def GetMedianValue(self):
        return np.median(self.deque)
        
    def GetAverageValue(self):
        return np.mean(self.deque)
        
    def GetGlobalAverageValue(self):
        return self.total / self.count
