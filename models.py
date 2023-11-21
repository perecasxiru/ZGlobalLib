import tensorflow as tf
import tensorflow_addons as tfa

from .callbacks import *
from .metrics import *

from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *

from tqdm.keras import TqdmCallback

class ZModel():
    """
    This wrapper defines basic properties of a Keras model.
    """
    def __init__(self, model):
        self.model = model
        
    @property
    def name(self):
        return self.model.name
    
    def compile(self, 
                num_classes,
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy', 
                metrics='auto',
                **kwargs):
        
        metrics=[tf.keras.metrics.CategoricalAccuracy(name=f'metrics/accuracy'),
                 tf.keras.metrics.TopKCategoricalAccuracy(3, name=f'metrics/top-3-accuracy'),
                 BalancedCategoricalAccuracy(num_classes, name='metrics/balanced-accuracy'),
                 tfa.metrics.F1Score(num_classes=num_classes, average='macro', name='metrics/F1-macro'),
                 tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes, name='metrics/AUC'),
                 tf.keras.metrics.Precision(name='metrics/precision'),
                 tf.keras.metrics.Recall(name='metrics/recall'),
                 tf.keras.metrics.PrecisionAtRecall(0.99, name='metrics/P@R_99'),
                 tf.keras.metrics.PrecisionAtRecall(0.95, name='metrics/P@R_95'),
                 tf.keras.metrics.PrecisionAtRecall(0.9, name='metrics/P@R_90'),
                 tfa.metrics.MatthewsCorrelationCoefficient(num_classes=num_classes, name='metrics/MCC')] if metrics=='auto' else metrics
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)
        
    def fit(self, *args,
            log_dir,
            patience=10,
            epochs=20,
            verbose=0,
            callbacks='auto', 
            validation_data=None,
            **kwargs):
        
        callbacks = [DuplicatedModelCheck(model_log_dir=log_dir),
                       SimpleLogger(log_dir=log_dir),
                       EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=patience),
                       MyReduceLROnPlateau('optimizer', monitor='val_loss', patience=patience//2, min_lr=1e-7),       
                       ModelCheckpoint(log_dir, monitor=f"val_loss", save_best_only=True, save_weights_only=True),
                       # ConfusionMatrixLogger(log_dir=log_dir, val_generator=val_generator, class_names=sorted(list(class_names))),
                       # UMAPLogger(log_dir, val_generator, class_names=sorted(list(class_names)), embedding_layer='efficientnetb3'),
                       TqdmCallback()] if callbacks=='auto' else callbacks
        self.model.fit(*args, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_data=validation_data, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, callbacks=[TQDMPredictCallback(leave=False)], **kwargs)
        

def get_model(name='EfficientNetB3', 
              input_shape=(256,256,3), 
              num_classes=6, 
              weights='imagenet', 
              data_augmentation=True, 
              last_embedding=None):
    
    if name == 'Simple':
        ef = Sequential()
        ef.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        ef.add(Conv2D(128, (3, 3), activation='relu'))
        ef.add(MaxPooling2D(pool_size=(2, 2)))
        ef.add(Dropout(0.25))

        ef.add(Conv2D(56, (3, 3), activation='relu', padding='same'))
        ef.add(Conv2D(56, (3, 3), activation='relu'))
        ef.add(MaxPooling2D(pool_size=(2, 2)))
        ef.add(Dropout(0.25))
        
        ef.add(Conv2D(12, (3, 3), activation='relu', padding='same'))
        ef.add(Conv2D(12, (3, 3), activation='relu'))
        ef.add(MaxPooling2D(pool_size=(2, 2)))
        ef.add(Dropout(0.25))

        ef.add(Flatten())
        ef.add(Dense(512, activation='relu'))
    
    if name == 'EfficientNetB0':
        ef = EfficientNetB0(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
        
    if name == 'EfficientNetB1':
        ef = EfficientNetB1(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
        
    if name == 'EfficientNetB2':
        ef = EfficientNetB2(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
    
    if name == 'EfficientNetB3':
        ef = EfficientNetB3(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
        
    if name == 'EfficientNetB4':
        ef = EfficientNetB4(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')

    if name == 'ResNet50':
        ef = ResNet50V2(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
        
    inputs = Input(shape=input_shape)
    
    x = inputs
    if data_augmentation:
        
        da_model = Sequential([
            RandomFlip("horizontal_and_vertical"),
            # RandomContrast(factor=0.2),
            # RandomBrightness(factor=0.2, value_range=(0.,1.)),
        ], name='DataAugmentation')
        
        x = da_model(x)
        
    ef_out = ef(x)
    
    if last_embedding is not None:
        ef_out = Dense(last_embedding, activation='relu', name='last_embedding')(ef_out)
    
    # ef_out = Dropout(0.5)(ef_out)
    outputs = Dense(num_classes, activation='softmax')(ef_out)

    model = ZModel(inputs, outputs)
    model._name = name
    return model