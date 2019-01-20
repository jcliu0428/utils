from ./ import is_str,is_list_of

file_handlers = {'json':JsonHandler(),
                 'yaml':YamlHandler(),
                 'yml':YamlHandler(),
                 'pickle':PickleHandler(),
                 'pkl':PickleHandler()
                 }
                 
def load(file,file_format = None,**kwargs):
    if file_format is None and is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('Unsupported format:{}'.format(file_format))
        
    handler = file_handlers[file_format]
    if is_str(file):
        obj = handler.load_from_path(file,**kwargs)
    elif hasattr(file,'read'):
        obj = handler.load_from_fileobj(file,**kwargs)
        
    else:
        raiseTypeError('file must be a filepath str or a file-object.')
    return obj
    
    
    
def dump(obj,file=None,file_format=None,**kwargs):
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError('file_format must be specified since file is None.')
    if file_format not in file_handlers:
        raise TypeError('Unsupported format {}'.format(file_format))
        
    handler = file_handlers[file_format]
    if file is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError('file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError('Unsupported format:{}'.format(file_format))
        
    handler = file_handler[file_format]
    if file is None:
        return handler.dump_to_str(obj,**kwargs)
    elif is_str(file):
        handler.dump_to_path(obj,file,**kwargs)
    elif hasattr(file,'write'):
        handler.dump_to_fileobj(obj,file,**kwargs)
    else:
        raise ValueError('file must be a filename str or a file-object.')
