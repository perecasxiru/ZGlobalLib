import cv2
import numpy as np
from tqdm.notebook import tqdm

def read_video(pth, as_array=True):
    fms = []
    vid = cv2.VideoCapture(pth)
    tbar = tqdm(leave=True)
    
    while True:
        ret, frame = vid.read()
        tbar.update(1)
        if not ret:
            break
        frame = np.flip(frame.astype(np.uint8), -1)
        fms.append(frame)
        
    tbar.close()
    
    if as_array:
        return np.array(fms)
    return fms