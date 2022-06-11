import os
from moviepy.editor import VideoFileClip

for filename in os.listdir('.'):
    name = filename.split('.')[0]
    ext = filename.split('.')[-1]
    if ext == 'mp4':
        if os.path.exists(name + '.gif'):
            continue
        videoClip = VideoFileClip(filename)
        videoClip.write_gif(filename.split('.')[0] + '.gif')