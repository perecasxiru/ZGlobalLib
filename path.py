import os
import glob
from datetime import datetime

def get_files(base_dir, ext=None, max_depth=None, exclude=None, 
              contains_any=None, contains_all=None):
    """
    Returns all the files of base_dir recursively
    
    Params
    ======
    :base_dir: Base path
    :ext: Extension (int or list) to filter
    
    Returns
    =======
    A list containing the desired files
    
    """
    
    assert (ext is None) or (type(ext) in [list, str])
    assert (exclude is None) or (type(exclude) in [list, str])
    assert (contains_any is None) or (type(contains_any) in [list, str])
    assert (contains_all is None) or (type(contains_all) in [list, str])
    
    ext = [ext] if type(ext)== str else ext
    exclude = [exclude] if type(exclude)== str else exclude
    contains_any = [contains_any] if type(contains_any)==str else contains_any
    contains_all = [contains_all] if type(contains_all)==str else contains_all
    
    ret_files = []
    
    search_pattern = f'{base_dir}/**' if max_depth is None else f'{base_dir}' + '/*' * max_depth
    for f in glob.glob(search_pattern, recursive=True):
        if (ext is not None) and not any([f.endswith(e) for e in ext]):
            continue
        if (exclude is not None) and any([e in f for e in exclude]):
            continue
        if (contains_any is not None) and not any([e in f for e in contains_any]):
            continue
        if (contains_all is not None) and not all([e in f for e in contains_all]):
            continue
        ret_files.append(f)
        
    return ret_files


def get_images(base_dir, ext=['bmp', 'png', 'jpg', 'jpeg'], 
               max_depth=None, exclude=None, contains_any=None, contains_all=None):
    return get_files(base_dir, ext=ext, max_depth=max_depth, 
                     exclude=exclude, contains_any=contains_any, contains_all=contains_all)

def get_log_dir(base_logdir='logs', *params):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base_logdir, now, *params)

def get_events(base_folder, contains='', sort_key=lambda x: x):
    files = get_files(base_folder, 'v2')    
    
    contains = contains if type(contains)==list else [contains]
    for pat in contains:
        files = [f for f in files if pat in f]
    
    assert len(files)>0
    return sorted(files, key=sort_key)

def get_val_events(base_folder, contains='', sort_key=lambda x: x):
    contains = contains if type(contains)==list else [contains]
    contains = ['/val/'] + contains
    return get_events(base_folder, contains, sort_key)

def get_train_events(base_folder, contains='', sort_key=lambda x: x):
    contains = contains if type(contains)==list else [contains]
    contains = ['/train/'] + contains
    return get_events(base_folder, contains, sort_key)