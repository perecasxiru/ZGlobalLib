import os
import io
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from shutil import rmtree
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.utils import plot_model                
from keras import backend as K
from sklearn.metrics import ConfusionMatrixDisplay

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class DuplicatedModelCheck(keras.callbacks.Callback):
    def __init__(self, model_log_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_log_dir = model_log_dir
        
        logs_dir = model_log_dir.split('/')[0]
        self._logs_dir = logs_dir if logs_dir[-1]=='/' else logs_dir+'/'
        
    def on_train_begin(self, logs=None):        
        # Remove logs prefix and date
        current_date, stp = self._model_log_dir[self._model_log_dir.find(self._logs_dir)+len(self._logs_dir):].split('/', 1)
        current_date = self._logs_dir + current_date
        
        # Available directories
        dirs = os.listdir(self._logs_dir)
        
        # Check all the dirs
        for d in dirs:
            # Check one dir
            path = self._logs_dir + d
            
            # Extract its date
            date = self._logs_dir + path[path.find(self._logs_dir)+len(self._logs_dir):].split('/', 1)[0]
            
            # If date is not the current_date (this check is necessary since the current log is created before this check)
            # And there is a model with the same name
            # print(path, stp, date, current_date)
            if (os.path.exists(path+'/'+stp)) and (date!=current_date):
                print(f"Model {stp} already in {path+'/'+stp}")
                
                # Choose between deleting or aborting
                select = widgets.Select(
                    options=[('-', None),
                             ('Exit and change the name', -1),
                             ('Remove and create new', -2),
                            ],
                    index=None,
                    disabled=False
                )
                output = widgets.Output()

                def on_change(change):
                    if change['type'] == 'change' and change['name'] == 'value':
                        idx = change['new']
                        if idx == -1:
                            pass
                            # rmtree(current_date)
                        if idx == -2:
                            rmtree(date)
                            # rmtree(current_date)
                        select.close()

                rmtree(current_date) # ALERT: Currently generated events file will be erased
                display(select, output)
                select.observe(on_change)

                raise Exception()
                
                
                
from tensorflow.python.platform import tf_logging as logging
from tensorflow.keras.callbacks import ReduceLROnPlateau

class MyReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer_name: str, *args, **kwargs):
        """
        It overrides the keras Callback ReduceLROnPlateau by letting you choose the optimizer to modify.
        
        Params
        ======
        :optimizer_name: The name of the optimizer the ReduceLROnPlateau callback will be applied to.
        
        """
        self.optimizer_name = optimizer_name
        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, epoch, logs=None):
        
        # Value to compare 
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
                
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = K.get_value(getattr(self.model, self.optimizer_name).lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(getattr(self.model, self.optimizer_name).lr, new_lr)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch +1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

class SimpleLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, show_model=True, expand_nested=True, 
                 optimizer_name='optimizer', lr_name='details/learning-rate', *args, **kwargs):
        """
        It logs some metrics to be displayed at the Tensorboard UI.
        
        Params
        ======
        :log_dir: The path of the log file.
        :show_model: Whether or not to store an image of the current model
        :expand_nested: If show_model is True, this will expand the submodels the model contains.
        :optimizer_name: String (for a single value) or List for a multiple values. It contains the name 
                         of the optimizers we want to log.
        :lr_name: String (for a single value) or List for a multiple values. For each optimizer in the 
                  :optimizer_name: list, a log will be created.        
        """
        super().__init__(*args, **kwargs)
        
        # To save the model architecture as a PNG image
        self.show_model = show_model
        self.expand_nested = expand_nested
        
        # The optimizer that have the learning rate we want to display.
        # TODO: Multiple learning rates
        self.optimizer_name = optimizer_name if type(optimizer_name) == list else [optimizer_name]
        self.lr_name = lr_name if type(lr_name) == list else [lr_name]
        
        # Writters for the metrics and values
        self.train_writer = tf.summary.create_file_writer(log_dir+'/train')
        self.val_writer = tf.summary.create_file_writer(log_dir+'/val')

        assert len(self.optimizer_name) == len(self.lr_name), 'Different amount of optimizers and names'
    
    def on_train_begin(self, logs=None):
        
        if not self.show_model:
            return
        
        mydpi = 200
        
        try:
            self.model.plot(expand_nested=self.expand_nested)
        except:
            plot_model(self.model, to_file='model.png', show_shapes=True, dpi=mydpi, expand_nested=self.expand_nested)
        
        im = imread('model.png')
        h, w, _ = im.shape
        fig = plt.figure(figsize=(w/mydpi, h/mydpi))
        ax = fig.gca()
        ax.imshow(im)
        ax.axis('off')
        plt.tight_layout()
        
        cm_image = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.train_writer.as_default():
            tf.summary.image("2. Metrics/Model", cm_image, step=0)
            self.train_writer.flush()
        
    
    def on_epoch_begin(self, epoch, logs=None):
        # Learning rate logger        
        for optimizer_name, lr_name in zip(self.optimizer_name, self.lr_name):
            model_optimizer = getattr(self.model, optimizer_name)
            try:
                lr = float(tf.keras.backend.get_value(model_optimizer.learning_rate))
            except:
                lr = float(tf.keras.backend.get_value(model_optimizer.lr(model_optimizer.iterations)))        

            for writer in [self.train_writer, self.val_writer]:
                with writer.as_default():
                    tf.summary.scalar(lr_name, data=lr, step=epoch)        
                    writer.flush()

    def on_epoch_end(self, epoch, logs=None):              
        # Log metrics
        for k,v in logs.items():
            writer = self.val_writer if k.startswith('val_') else self.train_writer
            name = k[4:] if k.startswith('val_') else k
            with writer.as_default():
                tf.summary.scalar(name, data=v, step=epoch)
                writer.flush()
    
    # This is only called in model.evaluate() Not true...
    # def on_test_batch_end(self, batch, logs=None):
    #     for k,v in logs.items():
    #         writer = self.val_writer # Overwrite the writer
    #         name = k[4:] if k.startswith('val_') else k
    #         with writer.as_default():
    #             tf.summary.scalar(name, data=v, step=0) # FIXME 0?
    #             writer.flush()
        
                               
class ConfusionMatrixLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, val_generator, class_names, 
                 moment='on_epoch_begin',
                 *args, **kwargs):
        """
        :moment: When to perform the callback action. One of ['on_epoch_begin', 'on_epoch_end', 'on_train_end']
        """
        
        super().__init__(*args, **kwargs)        
        self.writer = tf.summary.create_file_writer(log_dir+'/val')
        self.val_generator = val_generator
        self.class_names = class_names
        
        
        assert moment in ['on_epoch_begin', 'on_epoch_end', 'on_train_end']
        self.moment = moment
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.moment == 'on_epoch_begin':
            self.__function(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.moment == 'on_epoch_end':
            self.__function(epoch, logs)
    
    def on_train_end(self, logs=None):
        if self.moment == 'on_train_end':
            self.__function(0, logs)
    
    def __function(self, epoch, logs):
        _tmp_hist = self.model.history.history
        
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.val_generator)
        test_pred = np.argmax(test_pred_raw, axis=1)
        
        y_test = np.array([y for _, y in self.val_generator])
        test_labels = y_test.reshape(np.prod(y_test.shape[:2]), len(self.class_names)).argmax(1)

        # Calculate the confusion matrix.
        fig, ax = plt.subplots(figsize=(10,10))
        ConfusionMatrixDisplay.from_predictions(test_labels, test_pred, 
                                                display_labels=self.class_names, ax=ax, 
                                                xticks_rotation='vertical', normalize='true')
        plt.tight_layout()
        
        # Log the confusion matrix as an image summary.
        # figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.writer.as_default():
            tf.summary.image("1. Metrics/Confusion Matrix", cm_image, step=epoch)
            self.writer.flush()
            
        self.model.history.history = _tmp_hist
        
            

import umap
import umap.plot
import umap.aligned_umap
from tensorflow.keras.models import Model
class UMAPLogger(keras.callbacks.Callback):
    def __init__(self, log_dir, val_generator, class_names, embedding_layer=None, 
                 moment='on_epoch_begin',
                 *args, **kwargs):
        """
        Outputs a UMAP image per epoch
        
        Params
        ======
        :log_dir: The logation of the log file
        :val_generator:
        :class_names:
        :embedding_layer: Default is None (Last Layer of the model).
                          If a layer name is specified, it will use the embedding generated by the layer.
        
        """
        super().__init__(*args, **kwargs)        
        self.writer = tf.summary.create_file_writer(log_dir+'/val')
        self.val_generator = val_generator
        self.class_names = class_names
        self.embedding_layer = embedding_layer
        self.UMAP = None
        
        assert moment in ['on_epoch_begin', 'on_epoch_end', 'on_train_end']
        self.moment = moment
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.moment == 'on_epoch_begin':
            self.__function(epoch, logs)
    
    def on_epoch_end(self, epoch, logs=None):
        if self.moment == 'on_epoch_end':
            self.__function(epoch, logs)
    
    def on_train_end(self, logs=None):
        if self.moment == 'on_train_end':
            self.__function(0, logs)
    
    @staticmethod  
    def axis_bounds(embedding):
        left, right = embedding.T[0].min(), embedding.T[0].max()
        bottom, top = embedding.T[1].min(), embedding.T[1].max()
        adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
        return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]
    
    def __function(self, epoch, logs=None):
        if self.embedding_layer is not None:
            encoder = Model(self.model.inputs, self.model.get_layer(self.embedding_layer).get_output_at(0))
        else:
            encoder = self.model
        
        encoded = []
        labs = []
        for vg in self.val_generator:
            ims, lb = vg
            encoded.append(encoder.predict(ims))
            labs.append(lb)
        
        embedding_dim = len(encoded[0][0])
        encoded = np.array(encoded).reshape(-1, embedding_dim)
        labs = np.argmax(np.array(labs).reshape(-1, len(self.class_names)), 1)
        
        relations = {i:i for i in range(len(encoded))}
        if epoch == 0:
            self.UMAP = umap.AlignedUMAP().fit([encoded, encoded], relations=[relations])
        else:
            self.UMAP.update(encoded, relations=relations)

        fig, ax = plt.subplots(figsize=(10,10))
        ax_bound = self.axis_bounds(np.vstack(self.UMAP.embeddings_))
        
        cmap = plt.get_cmap('Spectral')
        mx = np.max(labs)
        
        for g in np.unique(labs):
            ix = np.where(labs == g)
            col = cmap(g/mx)
            c = [col]*len(ix)
            ax.scatter(*self.UMAP.embeddings_[-1][ix].T, c=c, label=self.class_names[g], s=25)
        
        ax.axis(ax_bound)
        ax.set(xticks=[], yticks=[])
        # plt.set_cmap('Spectral')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        # mapper = umap.UMAP().fit(encoded)
        # umap.plot.points(mapper, labels=np.array([self.class_names[l] for l in labs]), ax=ax);
        
        cm_image = plot_to_image(fig)

        # Log the confusion matrix as an image summary.
        with self.writer.as_default():
            tf.summary.image("1. Metrics/UMAP", cm_image, step=epoch)
            self.writer.flush()

            
from tqdm.notebook import tqdm
class TQDMPredictCallback(keras.callbacks.Callback):
    def __init__(self, custom_tqdm_instance=None, tqdm_cls=tqdm, leave=True, **tqdm_params):
        super().__init__()
        self.tqdm_cls = tqdm_cls
        self.tqdm_progress = None
        self.prev_predict_batch = None
        self.custom_tqdm_instance = custom_tqdm_instance
        self.tqdm_params = tqdm_params
        self.leave = leave

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        self.tqdm_progress.update(batch - self.prev_predict_batch)
        self.prev_predict_batch = batch

    def on_predict_begin(self, logs=None):
        self.prev_predict_batch = 0
        if self.custom_tqdm_instance:
            self.tqdm_progress = self.custom_tqdm_instance
            return

        total = self.params.get('steps')
        if total:
            total -= 1

        self.tqdm_progress = self.tqdm_cls(total=total, leave=self.leave, **self.tqdm_params)

    def on_predict_end(self, logs=None):
        if self.tqdm_progress and not self.custom_tqdm_instance:
            self.tqdm_progress.close()
            
            
from tensorflow.keras.callbacks import LearningRateScheduler     
class MyLearningRateScheduler(LearningRateScheduler):     
    
    def __init__(self, schedule):
        self.scheduler = schedule
        
        super().__init__(schedule)
    
    def plot(self, start_lr=1e-3, num_epochs=10, log_scale=True):
        s = self.scheduler(0, start_lr)
        a = [s]
        for i in range(1,num_epochs):
            s = self.scheduler(i, s)
            a.append(s)

        if log_scale:
            plt.yscale('log')
        plt.plot(a)
    
class LinearLearningRateScheduler(MyLearningRateScheduler):
    def __init__(self, wait=0, decrease=0.1):
        """
        It creates a linear decreasing learning rate.
        
        Params
        ======
        :wait: Wait some epochs before starting to decrease learing rate.
        :decrease: The amount of lr to reduce. new_lr = lr * decrease
        """
        self.wait = wait
        self.decrease = decrease
        
        super().__init__(schedule=self.scheduler)
    
    def scheduler(self, epoch, lr):
        if epoch < self.wait:
            return lr
        return lr * self.decrease
    
class ExpLearningRateScheduler(MyLearningRateScheduler):
    def __init__(self, wait=0, exp=0.1):
        """
        It creates a linear decreasing learning rate.
        
        Params
        ======
        :wait: Wait some epochs before starting to decrease learing rate.
        :exp: The amount of lr to reduce. new_lr = lr * e^(-exp)
        """
        self.wait = wait
        self.exp = exp
        
        super().__init__(schedule=self.scheduler)
    
    def scheduler(self, epoch, lr):
        if epoch < self.wait:
            return lr
        return lr * np.exp(-self.exp)
    
class StepLearningRateScheduler(MyLearningRateScheduler):
    def __init__(self, epochs, decrease=0.1):
        """
        It creates a step decreasing learning rate.
        
        Params
        ======
        :epochs: List of epochs after which the reduction is applied
        :decrease: The amount of lr to reduce. new_lr = lr * decrease. If a list, it has to be the same 
                   lenght as epochs, indicating the amount reduced in each epoch respectively.
        """
        self.epochs = epochs
        self.decrease = [decrease]*len(epochs) if type(decrease)!=list else decrease
        
        super().__init__(schedule=self.scheduler)
    
    def scheduler(self, epoch, lr):
        if epoch in self.epochs:
            return lr * self.decrease[np.argwhere(np.array(self.epochs)==epoch)[0][0]]
        return lr
    
    
        
import signal
class KeyboardInterruptCallback(keras.callbacks.Callback):
    def __init__(self):
        """
        It stops trainning when stop kernel is hit. The trainning stops at end of epoch.
        """
        super(KeyboardInterruptCallback, self).__init__()
        self.stopped_training = False

        # Register a signal handler for SIGINT (keyboard interrupt)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        print("\nTraining interrupted. Stopping training...")
        self.stopped_training = True

    def on_epoch_end(self, epoch, logs=None):
        if self.stopped_training:
            self.model.stop_training = True