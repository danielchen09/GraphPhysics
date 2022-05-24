import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
from PIL import Image
from tqdm import tqdm

from dataset import SwimmerDataset

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

def fig2array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    plt.close()
    return np.array(Image.frombytes("RGBA", (w, h), buf.tobytes()))

def draw_swimmer_from_graph(graph):
    draw_swimmer(graph.node_attrs)


def draw_swimmer(n_links, node_attrs, title=''):
    fig = plt.figure()
    for i in range(n_links):
        x = node_attrs[i, 0]
        y = node_attrs[i, 1]
        angle = node_attrs[i, 2]
        r = 0.05
        dy = np.cos(angle) * r
        dx = - np.sin(angle) * r
        plt.plot([x - dx, x + dx], [y - dy, y + dy], 'g', alpha = 0.5)
        plt.axis('equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(title)
    return fig

def generate_video(frames, name='test.mp4'):
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

if __name__ == '__main__':
    ds = SwimmerDataset(n_runs=1, shuffle=False, load_from_path=False)
    frames = []
    for graph, _ in tqdm(ds, desc='generating video'):
        fig = draw_swimmer(graph)
        buf = fig2array(fig)
        frames.append(buf)
    generate_video(frames)