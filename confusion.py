from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np
from .callbacks import TQDMPredictCallback

def confusion_from_generator(model, gen, class_names, check_generator=True, normalize=True):
    if check_generator:
        assert str(gen[0]) == str(gen[0]), 'Do not shuffle the generator'
    
    preds = model.predict(gen, callbacks=[TQDMPredictCallback(leave=True)])
        
    if gen is not None:
        labels = np.concatenate([l for _, l in gen])
    
    confusion_from_predictions(labels, preds, class_names)
    
    
def confusion_from_predictions(labels, preds, class_names, normalize=True):
    if len(preds.shape)>1:
        preds = preds.argmax(1)
    
    if len(labels.shape)>1:
        labels = labels.argmax(1)
    
    fig = plt.figure(figsize=(7,7))
    
    _norm = 'true' if normalize else None
    ConfusionMatrixDisplay.from_predictions(labels, preds, 
                                            display_labels=sorted(class_names), ax=fig.gca(), 
                                            xticks_rotation='vertical', normalize=_norm)
    plt.show()