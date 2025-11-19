from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy


TO_SECS = 1000


def observation_to_frame(obs):
  return (255 * obs.transpose((1, 2, 0))).astype('uint8')


def save_as_gif(frames, filepath, hz=10):
  if len(frames) > 0:
    hz = max(0, hz)
    pils = list()

    for idx, frame in frames:
      frame = Image.fromarray(frame.astype('uint8'))
      ImageDraw.Draw(frame).text((frame.size[0]//10, frame.size[1]//10),
                                str(idx),
                                fill=(0, 0, 0, 128),
                                font=ImageFont.load_default(size=frame.size[0]//10))
                                #font=ImageFont.truetype("sans-serif.ttf", size=frame.size[0]//10))
      pils.append(frame)

    if len(pils) > 0:
      pils[0].save(filepath, append_images=pils, save_all=True, loop=1,
                  duration=1 / hz * TO_SECS)

    # generate strip of frames
    if len(frames) < 15:
      for i in range(len(frames)+1, 16):
        frames.append((i, 0.1 * frames[0][1]))
    farray = [x[1] for x in frames[: -1: len(frames) // 14]]
    farray.append(frames[-1][1])
    inches = 8 if frames[0][1].shape[0] <= 64 else 16

    farray_strip = farray[::min(len(farray), 2)]
    if len(frames) % 14 != 0:
      farray_strip.append(farray[-1])
    farray_strip = numpy.concatenate(farray_strip, axis=1)
    plt.gcf().set_size_inches(1.5 * inches, inches)
    plt.gcf().set_dpi(2400)
    plt.gcf().add_subplot(1, 1, 1)
    plt.gca().imshow(farray_strip.astype('uint8'))
    plt.gca().axis('off')

    # plt.gca().legend(loc='upper right')
    plt.gcf().tight_layout(pad=0.2)
    plt.savefig(filepath.replace('.gif', '.png'), bbox_inches='tight')
    plt.gcf().clf()


#uncomment for coverting videos
#somehow moviepy is causing qt related errors in pycharm debugger
#  qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in 
#  "$HOME/anaconda3/envs/pgp/lib/python3.8/site-packages/cv2/qt/plugins" 
#  even though it was found.
# resolved by replacing opencv-python w opencv-python-headless
# https://github.com/NVlabs/instant-ngp/discussions/300#discussioncomment-3179213

"""
from os import path
import sys
from glob import glob
import moviepy.editor as editor
from PIL import Image

video_dir = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) == 3 else -1
if not path.exists(video_dir):
  print(f"{video_dir} doesn't exist")
  exit(1)

for gif_file in glob(f"{video_dir}/*.gif"):
  clip = editor.VideoFileClip(gif_file)
  filename = path.basename(gif_file).replace('.gif', '.mp4')
  clip.write_videofile(path.join(video_dir, filename))
  
  with Image.open(gif_file) as im:
      for i in range(0, im.n_frames, skip if skip > 0 else 1):
        im.seek(i)
        im.save(gif_file.replace('.gif', f'_{i}.png'))
      # save last frame
      im.seek(im.n_frames - 1)
      im.save(gif_file.replace('.gif', f'_{im.n_frames - 1}.png'))
"""
