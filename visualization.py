from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

try:
    from imageio.v3 import imread
except:
    from imageio import imread

import io
import cv2
import tensorflow as tf

def resize_img_aspect(image, new_height=512):
    height, width, _ = image.shape
    ratio = width / height

    new_width = int(ratio * new_height)

    image = cv2.resize(image, (new_width, new_height))
    return image
    
def fig_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3).numpy()    
    return image


def merge_image_mask(im, mk, alpha=0.3, channel='blue'):
    """
    Given an image and a mask of the same size, it blends the images together.
    """
    
    color2int = dict(zip(['red', 'green', 'blue'], range(3)))    
    assert channel in color2int
    
    mask_rgb = np.zeros_like(im)
    mask_rgb[:,:,color2int[channel]] = mk
    merge_image = (1 - alpha) * im + alpha * mask_rgb
    return merge_image

def plot_frames(images, columns=10, show_titles=True, figsize=None, title_fontsize=18, 
                savepath=None, save_height=512, display_fig=True):
    
    """
    Show frames from a given array of images.
    
    Params
    ======
    
    :images: Array of images or array of (image, title) tuples
    :columns: Number of images per row in the resulting visualization
    :show_titles: Whether or not to show the titles together with the images
    :figsize: Size of the resulting figure. Default is (35, nrows)
    :savepath: If not None, figure will be saved to this location
    :save_height: Height of the saved figure. Width will be resized to adapt the new height
    :display_fig: If savepath is not None, you can decide to plot or not the saved figure
    """
    
    if type(images) == zip:
        images = list(images) 
        
    assert len(images)>0, "No image to display" 
    
    # Check if images contain titles
    show_titles = show_titles if len(images[0])==2 else False
    # assert (not show_titles) or (show_titles and len(images[0])==2), "If show_titles is True, 'images' has to be an array of pairs (image, title)"
    
    columns = min(columns, len(images))
    nrows = len(images)//columns+int(len(images)%columns > 0)
    
    # Set default figsize if not provided
    figsize = (columns*5,nrows*5) if figsize is None else figsize
    fig, axs = plt.subplots(nrows, columns, figsize = figsize)   

    # This controls if only one value is passed
    if type(axs).__module__ != np.__name__:
        axs = np.array([axs])

    axs = axs.flatten()

    for idx_ax in range(len(images)):
        im2show = images[idx_ax][0] if len(images[idx_ax])==2 else images[idx_ax]
        
        if type(im2show) not in [np.array, np.ndarray]:
            im2show = imread(im2show)
            
        axs[idx_ax].imshow(im2show)
        axs[idx_ax].get_xaxis().set_visible(False)
        axs[idx_ax].get_yaxis().set_visible(False)
        
        if show_titles:
            axs[idx_ax].set_title(images[idx_ax][1], fontsize=title_fontsize)

    for id_ax in range(len(images), len(axs)):
        fig.delaxes(axs[id_ax])

    plt.tight_layout()
    
    if savepath:
        image = fig_to_img(fig)
        image = resize_img_aspect(image, save_height)
        plt.imsave(savepath, image)
    
    if display_fig:
        plt.show()
    else:
        plt.close(fig)
    
    
def plot_mosaic(ims, ttls=None, new_shape=1024, columns=5, orientation='horizontal'):
    """
    Plots images in a mosaic figure (no borders, no titles)
    
    Params
    ======
    
    :ims: Array of images to plot
    :ttls: Titles. If provided, they are printed following the figure shape before the plot
    :new_shape: Width of the resulting figure
    :columns: Number of images per row
    :orientation: 'horizontal' or 'vertical'.
                  If horizontal, images are plotted from left to right and from top to bottom.
                  If vertical, images are plotted from top to bottom and from left to right.
    """
    
    # Check orientation
    assert orientation in ['horizontal', 'vertical']
    
    # Compute number of rows and columns.
    # if orientation == 'horizontal':
    nrows, ncols = len(ims)//columns + 1 if len(ims)%columns!= 0 else len(ims)//columns, len(ims) if len(ims)<columns else columns
    big_im = Image.new('RGB', (512*ncols, 512*nrows))
    
    for num, im in enumerate(ims):
        if orientation == 'horizontal':
            big_im.paste(Image.fromarray(im), ((num%columns)*512, (num//columns)*512))
        else:
            big_im.paste(Image.fromarray(im), ((num//nrows)*512, (num%nrows)*512))
    
    if ttls is not None:
        # assert , 'Titles not correct in horizontal orientation'
        ttls = ttls + ["" for _ in range(0, ncols*nrows - len(ims))]
        ttls = ttls if orientation == 'horizontal' else [eval(a) for a in np.array([str(a) for a in ttls]).reshape(ncols, nrows).T.flatten()]
        ttls = ttls[:len(ims)]
        for i in range(0, len(ims), columns):
            print(ttls[i:i+columns])
        
    new_shape2 = int(new_shape*(nrows/ncols))
    return big_im.resize((new_shape, new_shape2)).show()