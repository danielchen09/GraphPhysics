import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
from PIL import Image
from tqdm import tqdm
import math

import config


def q2e(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    a = -np.pi - np.array([roll_x, pitch_y, yaw_z]) # in radians
    # a = a[np.where(a % np.pi != 0)]
    # return a[0] if len(a) > 0 else 0
    return a


def fig2array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    plt.close()
    return np.array(Image.frombytes("RGBA", (w, h), buf.tobytes()))

def draw_swimmer_from_graph(graph):
    draw(graph.node_attrs)

def draw(n_links, node_attrs, title=''):
    fig = plt.figure()
    for i in range(n_links):
        x = node_attrs[i, config.DRAW_X_IDX,]
        y = node_attrs[i, config.DRAW_Y_IDX]
        angle = q2e(*node_attrs[i, 3:7].detach().numpy())[config.ANGLE_IDX]
        r = config.BODY_LENGTH
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        plt.plot([x - dx, x + dx], [y - dy, y + dy], 'g', alpha = 0.5)
        plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(title)
    return fig

def generate_video(frames, name='test.mp4'):
    if config.LINUX:
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=1000 / 30, blit=True, repeat=False)
    anim.save(name)