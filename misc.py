import collections
import six

def is_str(x):
    return isinstance(x,six.string_types)

def is_seq_of(seq,expected_type,seq_type=None):
    """Args:seq(Sequence):The Sequence to be checked.
            expected_type:Expected type of sequence items.
            seq_type(type,optional):Expected sequence type.
       Returns:
            bool:Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = collections.Sequence
    else:
        assert isinstance(seq_type,type)
        exp_seq_type = seq_type
    if not isinstance(seq,exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item,expected_type):
            return False
    return True


def is_list_of(seq,expected_type):
    return is_seq_of(seq,expected_type,seq_type=list)
    
def is_tuple_of(seq,expect_type):
    return is_seq_of(seq,expected_type,seq_type=tuple)
    
def concat_list(in_list):
    return list(itertools.chain(*in_list))
    
def slice_list(in_list,lens):
    """Args:in_list(list):The list to be sliced.
            lens(int or list):The expected length of each out list.
       Returns:
           list:A list of sliced list.
    """
    if not isinstance(lens,list):
        raise TypeError('indices' must be a list of integars)
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not match:{} != {}'.format(sum(lens),len(in_list)))
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx+lens[i]])
        idx += lens[i]
    return out_list
