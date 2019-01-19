import six
def iter_cast(inputs,dst_type,return_type=None):
    """dst_type is the destination type of each element of this iterator.
       return_type is the destination type of this iterator,
       if return_type == None,just return a iterator.
    """

    if not isinstance(inputs,collections.Iterable):
        raise TypeError('Inputs must be an iterable object.')
    if not isinstance(dst_type,type):
        raise TypeError('dst_type must be a valid type')
    
    out_iterable = six.moves.map(dst_type,inputs)
    
    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)
        
def list_cast(inputs,dst_type):
    return iter_cast(inputs,dst_type,return_type=list)
    
def type_cast(inputs,dst_type):
    return iter_cast(inputs,dst_type,return_type=tuple)
