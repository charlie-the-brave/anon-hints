import os
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt


# HEADLESS RENDERING
# xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 vis_track.py

plt.rcParams.update({"font.size": 18})
dlt = 20


def find_hard_corners(track):
  """ snipped from python3.8/site-packages/gym/envs/box2d/car_racing.py """
  TRACK_TURN_RATE = 0.31
  BORDER_MIN_COUNT = 4
  # Red-white border on hard turns
  border = [False] * len(track)
  for i in range(len(track)):
      good = True
      oneside = 0
      for neg in range(BORDER_MIN_COUNT):
          beta1 = track[i - neg - 0][1]
          beta2 = track[i - neg - 1][1]
          good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
          oneside += np.sign(beta1 - beta2)
      good &= abs(oneside) == BORDER_MIN_COUNT
      border[i] = good
  for i in range(len(track)):
      for neg in range(BORDER_MIN_COUNT):
          border[i - neg] |= border[i]

  return border


def vis_track(env, id, outpath, track_pos=None):
  mask = find_hard_corners(env.track)

  # compute corner labels { (x,y,c) }, where track[i] = (_, heading, x, y)
  sharp_corners = []
  R, L = 2, len(env.track)
  for t in range(R, L - R - 1):
    c, x, y, tx, ty, l = env.compute_cues(t)[:6]
    if c > 0.20:
      sharp_corners.append([x, y, c, tx, ty, l])

  X, Y = [t[2] for t in env.track], [t[3] for t in env.track]
  X, Y = np.array(X), np.array(Y)

  K = 2
  plt.gcf().set_size_inches(10,8)
  plt.gcf().set_dpi(600)
  plt.gcf().add_subplot(K,1,1)
  # show gt corners
  plt.gca().set_facecolor('xkcd:grass green')
  grid_x, grid_y = np.meshgrid(np.linspace(-155,255, 10), np.linspace(-155,155, 5))
  plt.gca().scatter(grid_x, grid_y, s=1024, marker='s', c='xkcd:green', zorder=-1)
  plt.gca().plot(X, Y, linewidth=15, linestyle='solid', c='grey', zorder=-1)
  plt.gca().scatter(X[0], Y[0], s=840, marker='*', label='start', c='yellow', zorder=1)
  plt.gca().scatter(X[mask], Y[mask], s=220, marker='s', c='red', label='corners in sim', zorder=1)
  plt.gca().set_ylim([-155,155])
  plt.gca().set_xlim([-155,255])
  plt.gca().legend(loc='upper right')
  [axps.set_visible(False) for axps in plt.gca().spines.values()]
  plt.gca().axes.get_xaxis().set_visible(False)
  plt.gca().axes.get_yaxis().set_visible(False)
  # show computed corners
  plt.gcf().add_subplot(K,1,2)
  plt.gca().plot(X, Y, linewidth=5, linestyle='dashed', c='grey')
  for i, tt in enumerate(sharp_corners):
    if i == 0:
      plt.gca().scatter(tt[0], tt[1], s=60, marker='s', c='blue', label='corners from state')
      plt.gca().plot([tt[0] - dlt*tt[3], tt[0] + dlt*tt[3]], [tt[1] - dlt*tt[4], tt[1] + dlt*tt[4]], linewidth=1, linestyle='solid', c='red', label='corner tangents')
    else:
      plt.gca().scatter(tt[0], tt[1], s=60, marker='o', c='blue')
      plt.gca().plot([tt[0] - dlt*tt[3], tt[0] + dlt * tt[3]], [tt[1] - dlt*tt[4], tt[1] + dlt * tt[4]], linewidth=1, linestyle='solid', c='red')

  plt.gca().set_ylim([-155,155])
  plt.gca().set_xlim([-155,255])
  plt.gca().legend(loc='upper right')
  [axps.set_visible(False) for axps in plt.gca().spines.values()]
  plt.gca().axes.get_xaxis().set_visible(False)
  plt.gca().axes.get_yaxis().set_visible(False)

  # save visualisation
  plt.gcf().tight_layout(pad=0.2)
  plt.savefig(os.path.join(outpath, f'track_vis_{id}.png'), bbox_inches='tight')
  plt.gcf().clf()

  ## show cues (optional) and track positions
  if track_pos is not None:
    plt.gcf().set_size_inches(10,4)
    plt.gcf().set_dpi(600)
    plt.gcf().add_subplot(1,1,1)
    # show gt corners
    U, V = [t[0] for t in track_pos], [t[1] for t in track_pos]
    U, V = np.array(U), np.array(V)
    plt.gca().set_facecolor('xkcd:grass green')
    grid_x, grid_y = np.meshgrid(np.linspace(-155,255, 10), np.linspace(-155,155, 5))
    plt.gca().scatter(grid_x, grid_y, s=1024, marker='s', c='xkcd:green', zorder=-1)
    plt.gca().plot(X, Y, linewidth=30, linestyle='solid', c='grey', zorder=-1)
    plt.gca().scatter(U, V, s=20, marker='o', label='car pos', c='red', zorder=1)
    plt.gca().set_ylim([-155,155])
    plt.gca().set_xlim([-155,255])
    plt.gca().legend(loc='upper right')
    [axps.set_visible(False) for axps in plt.gca().spines.values()]
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # save visualisation
    plt.gcf().tight_layout(pad=0.2)
    plt.savefig(os.path.join(outpath, f'progress_vis_{id}.png'), bbox_inches='tight')
    plt.gcf().clf()


def vis_track_and_frame(z_type, env, id, outpath):
  vis_track(env, id, outpath)

  # save sample image after several sim steps
  env.reset()
  for i in range(int(np.random.uniform(120,180))):
    env.step(np.array([np.random.uniform(-0.1,0.1), 0.05, 0]))

  cues = np.array(env.compute_cues())
  frame = env.draw_cues(z_type, cues, annotate=True)

  Image.fromarray(frame.astype('uint8')).save(os.path.join(outpath, f'track_vis_frame_{id}.png'))


if __name__ == '__main__':
  import sys
  from env.envs import make_env
  from utils.config import load_config

  if len(sys.argv) != 2:
    print("usage\n\t vis_track.py <config filepath>")
    exit(-1)

  config_file = sys.argv[1]
  if not os.path.exists(config_file):
    print(f"'{config_file}' doesn't exist!")
    exit(-1)

  cfg = load_config(config_file)
  outpath = os.path.dirname(config_file)
  for i in range(10):
    env = make_env(cfg)
    env.reset(seed=int(np.random.randint(i,255)))
    vis_track_and_frame(cfg['z_type'],env, i, outpath)
