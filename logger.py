import logging
import datetime
import subprocess
import os
import io
import tensorflow as tf
import matplotlib.pyplot as plt

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

def fig_to_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return tf.image.decode_png(buf.getvalue(), channels=3).numpy()
    
class Logger():
    def __init__(self, log_dir='logfiles', log_name='Custom Logger', log_link='zz_latest_log'):
        """
        Params
        ------
        :log_dir: Directory where the log will be saved
        :log_name:
        
        """
        self.log_dir = log_dir
        self.log_link = log_link
        
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG) # better to have too much log than not enough
        logger.handlers.clear()
        logger.addHandler(self.get_file_handler(log_dir=log_dir))
        logger.propagate = False
        self.logger = logger
        self.create_link()
    
    def get_file_handler(self, log_dir='.'):
        # log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.join(log_dir, "_".join(log_dir.split('/')))
        
        file_handler = logging.handlers.TimedRotatingFileHandler(log_dir+'.log', when='midnight', encoding='utf-8')
        file_handler.setFormatter(FORMATTER)
        return file_handler

    def log(self, text, bl=False):
        self.logger.info(text)     
        if bl:
            self.logger.info('')  
            
    def log_from_dict(self, dct):
        for log_key, log_val in dct.items():
            self.log(f"{log_key}: {log_val}")
            
    def create_link(self):
        os.makedirs(self.log_dir, exist_ok=True)
        subprocess.run(["rm", "-f", self.log_link])
        subprocess.run(["ln","-sf", self.log_dir, self.log_link])
        
    def log_figure(self, fig, name):
        img_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        image = fig_to_img(fig)
        if '.' not in name:
            name += '.png'
        plt.imsave(os.path.join(img_dir, name), image)
        
        