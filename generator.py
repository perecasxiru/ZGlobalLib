import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import imageio
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import RandomCrop, Resizing

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
        
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def mask_multiple_images(images, mask_value=[255,0,0], radius=None):
    h, w, _ = images[0].shape
    mask = create_circular_mask(h, w, radius=radius)
    
    def map_fun(a):
        a[~mask] = mask_value
        return a

    return np.array(list(map(map_fun, images)))


class DataGenerator(keras.utils.Sequence) :
  
    def __init__(self, images, labels, batch_size, from_files=True,
                 mask_images=False, mask_radius=None, mask_color=[0,0,0], 
                 normalize255=False, onehot=False, 
                 resize=None, num_classes=None, from_files_start=False,
                 shuffle_epoch=True,
                ):        
        """
        Default generator to train mini-batches
        Params
        ======
        :images, labels:
        
        Returns
        =======
        :images, labels: with size batch_size.
        :from_files_start: If true, images are expected to be path files and they are loaded at start.
        """
        self.batch_size = batch_size
        self.from_files = from_files if not from_files_start else False
        self.mask_images = mask_images
        self.mask_radius = mask_radius
        self.mask_color = mask_color
        self.resize = resize
        
        self.shuffle_epoch = shuffle_epoch
        
        if from_files_start:
            self.images = np.array([imageio.imread(im) for im in images])
        else:
            self.images = images
            
        self.labels = labels
        
        self.normalize255 = normalize255
        self.onehot = onehot
        self.num_classes = num_classes if num_classes else len(np.unique(self.labels)) if self.onehot else None            
    
    def __len__(self) :
        _len = len(self.images)//self.batch_size
        return _len # FIXME: (If we do this, then .predict doesn't work) if len(self.images)%self.batch_size == 0 else _len+1
  
    def __getitem__(self, idx) :
        
        if (idx == 0) and (self.shuffle_epoch):
            
            # Shuffle at first batch
            c = list(zip(self.images, self.labels))
            random.shuffle(c)
            self.images, self.labels = zip(*c)
            self.images, self.labels = np.array(self.images), np.array(self.labels)       
        
        images = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
        labels = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        if self.from_files:
            images = np.array([imageio.imread(im) for im in images])
        
        if self.mask_images:
            images = mask_multiple_images(images, mask_value=self.mask_color, radius=self.mask_radius)
        
        if self.normalize255:
            images = images/255
            
        if self.resize:
            shp = images[0].shape[0] # Get the shape of an image. If not the same as the proposed, resize it
            if shp != self.resize:
                images = keras.layers.Resizing(self.resize, self.resize)(images).numpy()
            
        if self.onehot:
            labels = to_categorical(labels, num_classes=self.num_classes)
        return images, labels