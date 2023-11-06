from sklearn.utils import class_weight
import numpy as np
from tqdm.notebook import tqdm

def compute_class_weight(gen=None, labels=None):
    
    assert (labels is not None) or (gen is not None)
    
    # Check if labs are one-hot encoded
    if gen is not None:
        labels = np.concatenate([l.argmax(1) if len(l.shape)>1 else l for _, l in tqdm(gen, leave=False)])

    if len(labels.shape)>1:
        labels = labels.argmax(1)
        
    class_weights = class_weight.compute_class_weight('balanced',
                                                     classes=np.unique(labels),
                                                     y=list(labels))
    return dict(enumerate(class_weights))