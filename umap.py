import umap
import umap.plot
import numpy as np
from matplotlib import pyplot as plt

def plot_embeddings(embeddings, labels=None, 
                    color_key_cmap='Paired', figsize=(10,10), **kwargs):
    
    if type(labels)==list:
        labels = np.array(labels)
        
    ax = plt.figure(figsize=figsize).gca()
    # ax.axis('off')
    
    ax.set_xticks([])
    ax.set_yticks([])

    mapper = umap.UMAP().fit(embeddings)
    umap.plot.points(mapper, labels=labels, color_key_cmap=color_key_cmap,ax=ax, **kwargs)
