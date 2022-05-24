from datetime import datetime
from tensorboard import program
import tensorflow as tf

class Logger:
    def __init__(self, port=6006):
        self.log_dir = './logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer(self.log_dir, flush_millis=1000)
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir, '--reload_interval', '1', '--port', f'{port}'])
        self.url = tb.launch()
        print(f"Tensorflow listening on {self.url}")
        self.steps = 0

    def log_loss(self, name, value):
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)

    def step(self):
        self.steps += 1