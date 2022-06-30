import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import animation
from PIL import Image
from tqdm import tqdm
import math
from dm_control import mjcf
from dm_control import mujoco
import torch
import copy

import config

geom_type = {
    0: 'plane',
    2: 'sphere',
    3: 'capsule',
    4: 'ellipsoid',
    5: 'cylinder',
    6: 'box',
    7: 'mesh'
}

class Renderer:
    def __init__(self):
        self.arena = mjcf.RootElement()
        chequered = self.arena.asset.add('texture', type='2d', builtin='checker', width=300,
                                    height=300, rgb1=[.2, .3, .4], rgb2=[.3, .4, .5])
        grid = self.arena.asset.add('material', name='grid', texture=chequered,
                            texrepeat=[5, 5], reflectance=.2)
        self.arena.worldbody.add('geom', name='floor', type='plane', size=[2, 2, .1], pos=[0, 0, 0], material=grid)
        self.arena.worldbody.add('light', name='light0', pos=[0, 0, 5], dir=[0, 0, -1])

        self.arena.worldbody.add('camera', name='cam0', pos=[0, 0, 0], euler=[0, 0, 0])
        self.center = 0
        self.init = False

    def render(self, node_attrs, env_creator, env_key):
        # breakpoint()
        geom_names = copy.deepcopy(env_creator.geom_names[env_key])
        # ground_idx = geom_names.index(next(gn for gn in geom_names if gn in ['ground', 'floor']))
        # node_attrs = torch.cat([node_attrs[:ground_idx], node_attrs[ground_idx+1:]], dim=0)
        # geom_names.pop(ground_idx)
        sizes = env_creator.sizes[env_key]
        camera_config = env_creator.camera_configs[env_key]

        for i in range(len(geom_names)):
            g_name = geom_names[i]
            if geom_names[i] in ['ground', 'floor']:
                g_name = 'floor'
                node_attrs[i, 0:3] = 0
                node_attrs[i, 3] = 1
                node_attrs[i, 4:7] = 0
            pos = node_attrs[i, 0:3]
            quat = node_attrs[i, 3:7]
            size = sizes[i]
            body = self.arena.worldbody.find('geom', g_name)
            if body is not None:
                body.pos = pos
                body.quat = quat
                body.size = size
            else:
                self.arena.worldbody.add('geom', name=geom_names[i], type='capsule', size=size, pos=pos, quat=quat, rgba=[1, 0, 0, 1])
        physics = mjcf.Physics.from_mjcf_model(self.arena)
        cam = mujoco.MovableCamera(physics)

        self.center = node_attrs[:, 0:3].mean(axis=0).numpy()
        cam.set_pose(self.center, *camera_config)
        return cam.render()

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
    return a[config.RENDER_CONFIG.DRAW_ANGLE_IDX]

def fig2array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    plt.close()
    return np.array(Image.frombytes("RGBA", (w, h), buf.tobytes()))

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