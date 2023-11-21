import os
import glob
from datetime import datetime

def get_files(base_dir, ext=None):
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
    
    assert type(ext) in [list, str]
    ext = [ext] if type(ext)!= list else ext
    ret_files = []
    
    for f in glob.glob(f'{base_dir}/**', recursive=True):
        if (ext is not None) and not any([f.endswith(e) for e in ext]):
            continue
        ret_files.append(f)
        
    return ret_files


def get_images(base_dir, ext=['bmp', 'png', 'jpg', 'jpeg']):
    return get_files(base_dir, ext=ext)

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