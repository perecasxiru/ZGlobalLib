import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.utils import to_categorical

def loader(_class, shuffle=True, max_classes=None, new_on_eval=False, normalize=True, onehot=False):
    
    (train_images, train_labels), (val_images, val_labels) = _class.load_data()
    
    if shuffle:
        ptr = np.random.permutation(len(train_images))
        pte = np.random.permutation(len(val_images))
        
        train_images, train_labels, val_images, val_labels = train_images[ptr], train_labels[ptr], val_images[pte], val_labels[pte]

    train_labels, val_labels = train_labels.flatten(), val_labels.flatten()
    if normalize:
        train_images, val_images = train_images/255, val_images/255
        
    
    if max_classes is not None:
        
        if isinstance(max_classes, int):
            max_classes = list(range(max_classes))
        train_idxs = np.argwhere(np.isin(train_labels, max_classes))
        train_images, train_labels = np.squeeze(train_images[train_idxs]), np.squeeze(train_labels[train_idxs])
        
        if not new_on_eval:
            val_idxs = np.argwhere(np.isin(val_labels, max_classes))
            val_images, val_labels = np.squeeze(val_images[val_idxs]), np.squeeze(val_labels[val_idxs])
    
    if onehot:
        train_labels, val_labels = to_categorical(train_labels), to_categorical(val_labels)
        
    return train_images, train_labels, val_images, val_labels


def cifar_loader(shuffle=True, max_classes=None, cifar=10, new_on_eval=False, normalize=True, onehot=False):
    """
    It loads the CIFAR10 or CIFAR100 dataset.
    
    Params
    ======
    
    :shuffle: If images are shuffled 
    :max_classes: If <int>, max number of classes to use in the default order, e.g. 3 -> [0,1,2]
                  If <list>, desired classes to be kept.
                  
    Returns
    =======
    :train_images, train_labels, val_images, val_labels:
    """
    
    assert cifar in [10, 100]
    
    cifar_gen = keras.datasets.cifar10 if cifar==10 else keras.datasets.cifar100
    return loader(cifar_gen, shuffle=shuffle, max_classes=max_classes, new_on_eval=new_on_eval, normalize=normalize, onehot=onehot)


def mnist_loader(shuffle=True, max_classes=None, new_on_eval=False, normalize=True, repeat_channels=True, onehot=False):
    mnist = keras.datasets.mnist
    train_images, train_labels, val_images, val_labels = loader(mnist, shuffle=shuffle, max_classes=max_classes, new_on_eval=new_on_eval, normalize=normalize, onehot=onehot)
    
    if repeat_channels:
        train_images, val_images = np.stack([train_images, train_images, train_images], axis=-1), np.stack([val_images, val_images, val_images], axis=-1)
    return train_images, train_labels, val_images, val_labels