import tensorflow as tf

def use_gpu(gpu_idx=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
            _ = tf.config.experimental.list_logical_devices('GPU')
            return gpus[gpu_idx]
        except RuntimeError as e:
            print(e)