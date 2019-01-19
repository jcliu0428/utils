from .misc import is_str

def to_tensor(data):
    """Convert objects of various python types to torch.Tensor"""
    if isinstance(data,torch.Tensor):
        return data
    elif isinstance(data,np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data,Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isintance(data,int):
        return torch.LongTensor([data])
    elif isinstance(data,float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be transformed into tensor type.'.format(type(data)))
