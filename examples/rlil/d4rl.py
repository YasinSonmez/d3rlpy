import h5py
import numpy as np

from examples.rlil.visualize import render_episode

hf = h5py.File("hopper_medium-v2.hdf5", "r", locking=False)

obs = hf["observations"][:]  # shape (N, obs_dim)
actions = hf["actions"][:]  # shape (N, act_dim)
rews = hf["rewards"][:]  # shape (N,)
dones = hf["dones"][:]  # shape (N,)  ← “env done” flags
timeouts = hf["timeouts"][:]  # shape (N,)  ← “timeout” flags

hf.close()

# Get episodes
terminal_inds = np.where(np.logical_or(dones, timeouts))[0]
starts = np.concatenate([[0], terminal_inds[:-1] + 1])
ends = terminal_inds + 1
episodes = list(zip(starts, ends))
print(f"Found {len(episodes)} episodes")

lengths = [end - start for start, end in episodes]
returns = [rews[start:end].sum() for start, end in episodes]

# Replay
import gym

from d3rlpy.dataset.components import Episode

# Create the environment
env = gym.make("Hopper-v2", render_mode="rgb_array")
# Build an Episode object from HDF5 data for the first episode
start, end = episodes[0]
episode = Episode(
    observations=obs[start:end],
    actions=actions[start:end],
    rewards=rews[start:end],
    terminated=bool(dones[end - 1]),
)
# Create output directory for videos
video_dir = "videos"
import os

os.makedirs(video_dir, exist_ok=True)
# Define output path with d4rl in filename
video_path = os.path.join(video_dir, "d4rl_hopper_medium_v2.mp4")
# Render and save the episode
render_episode(episode, env, video_path)
print(f"Saved replay video to {video_path}")
