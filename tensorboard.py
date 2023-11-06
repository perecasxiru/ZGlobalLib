import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import numpy as np

def read_tfevent(path_to_events_file):
    dct = defaultdict(list)    
    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            dct[v.tag].append(tf.make_ndarray(v.tensor))
    for val in dct:
        dct[val] = np.array(dct[val])
    return dct


def merge_events(events, keep_val='all', mode='mean'):
    """
    Given a list of tensorboard events, returns a dictionary aggregating them in the specified format
    
    Params
    ======
    :events: list of tf events
    :keep_val: What to keep from each event. 
            :all: Keep all the curve
            :last: Keep only the last value of the curve
            :best: Keep only the best value of the curve
    :mode: function to use as aggregator: min, max or mean
    """
    assert keep_val in ['all', 'last', 'best']
    assert mode in ['min', 'max', 'mean']
    assert len(events)>1
    
    keys = set(events[0].keys())
    for e in events[1:]:
        keys.intersection_update(set(e.keys()))
    
    ret_metrics = {}
    # min_len = min([len(e[k]) for e in events for k in keys])
    for k in keys:
        new_array = []
        for e in events:
            to_append = e[k] if keep_val=='all' else e[k][-1:] if keep_val=='last' else [max(e[k])]
            new_array.append(to_append)
        func = np.mean if mode=='mean' else np.max if mode=='max' else np.min
        new_array = func(np.array(new_array), axis=0)
        ret_metrics[k] = new_array
        
    return ret_metrics


def concat_events(events):
    """
    It concatenates two or more dictionaries containing curves.
    """
    assert len(events)>1
    
    keys = set(events[0].keys())
    for e in events[1:]:
        keys.intersection_update(set(e.keys()))
    
    ret_metrics = {}
    # min_len = min([len(e[k]) for e in events for k in keys])
    for k in keys:
        new_array = []
        for e in events:
            new_array.append(e[k])
        new_array = np.concatenate(new_array)
        ret_metrics[k] = new_array
        
    return ret_metrics