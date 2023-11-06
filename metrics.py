import tensorflow as tf
import numpy as np

class BalancedCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='metrics/balanced-accuracy', **kwargs):
        """
        It computes the balanced accuracy by taking the accuracy of each individual class. The reported
        value is the mean of all these accuracies. It is useful for imbalanced datasets since using 
        traditional accuracy is biased for classes 'easy' to predict
        
        Params
        ======
        :num_classes: Number of different classes in the training set.        
        """
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name='cm', initializer='zeros', shape=(num_classes,num_classes))
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        a = tf.argmax(y_true, axis=-1)
        b = tf.argmax(y_pred, axis=-1)
        C = tf.cast(tf.math.confusion_matrix(a, b, num_classes=self.num_classes), self.dtype)
        self.confusion_matrix.assign_add(C)

    def result(self):
        C = self.confusion_matrix
        _sum = tf.cast(tf.reduce_sum(C, axis=1), self.dtype) + tf.keras.backend.epsilon()
        perclass = tf.cast(tf.linalg.tensor_diag_part (C), self.dtype) / _sum
        bac = tf.cast(tf.reduce_mean(perclass), self.dtype)
        return bac

    def reset_state(self):
        C = tf.cast(tf.constant(np.zeros((self.num_classes,self.num_classes))), self.dtype)
        self.confusion_matrix.assign(C)