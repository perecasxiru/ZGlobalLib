import tensorflow_addons as tfa
import tensorflow as tf

def OneHotTripletSemiHardLoss(**kwargs):
    
    def loss(y_true, y_pred):
        tl = tfa.losses.TripletSemiHardLoss(**kwargs)
        return tl(tf.argmax(y_true, 1), y_pred)
    
    return loss


def OneHotTripletHardLoss(**kwargs):
    
    def loss(y_true, y_pred):
        tl = tfa.losses.TripletHardLoss(**kwargs)
        return tl(tf.argmax(y_true, 1), y_pred)
    
    return loss