import argparse
import os
from itertools import zip_longest

import imageio
import numpy as np

import d3rlpy

os.environ["MUJOCO_GL"] = "egl"


def rollout_frames(env, episode, max_steps=1000):
    """Run one rollout, return list of rgb frames."""
    frames = []
    n_steps = min(episode.actions.shape[0], max_steps)

    obs, info = env.reset()

    qpos = np.zeros(env.model.nq)
    qpos[1:] = episode.observations[0, :5]
    qvel = episode.observations[0, 5:].copy()
    env.set_state(qpos, qvel)

    for i in range(n_steps - 1):
        # get the action from your policy
        obs, _, _, _, info = env.step(episode.actions[i])
        qpos = np.zeros(env.model.nq)
        qpos[1:] = episode.observations[i + 1, :5]
        qpos[0] = info["x_position"]
        qvel = episode.observations[i + 1, 5:]
        env.set_state(qpos, qvel)

        # render and store the RGB frame
        frame = env.render()
        frames.append(frame)

    return frames


def render_episode(episode, env, output_path, fps=30, max_frames=None):
    writer = imageio.get_writer(output_path, fps=fps)

    frames = rollout_frames(env, episode)
    for f in frames:
        writer.append_data(f)

    writer.close()
    env.close()


def render_episodes_grid_video(
    interval_episodes, env, output_path, fps=30, max_frames=None
):
    # Make the video from list of list of frames
    # Each list in interval_episodes corresponds to a row of rollouts.
    # Each rollout should be properly padded
    fps = 30
    writer = imageio.get_writer(output_path, fps=fps)

    frames_grid = [
        [rollout_frames(env, episode) for episode in row_episodes]
        for row_episodes in interval_episodes
    ]

    len(frames_grid)
    nc = max(len(row) for row in frames_grid)

    # padding
    h, w, c = frames_grid[0][0][0].shape
    blank = np.zeros((h, w, c), dtype=np.uint8)

    for row in frames_grid:
        while len(row) < nc:
            row.append([])

    row_iters = [zip_longest(*row, fillvalue=blank) for row in frames_grid]

    for row_frames in zip_longest(*row_iters, fillvalue=(blank,) * nc):
        # row_frames is an nr tuple, each elem has nc frames
        row_images = [np.concatenate(frames, axis=1) for frames in row_frames]
        grid_frame = np.concatenate(row_images, axis=0)
        writer.append_data(grid_frame)

    writer.close()
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize episodes by rewards"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mujoco/hopper",
        help="Minari environment name",
    )
    parser.add_argument(
        "--expert-level",
        type=str,
        default="expert-v0",
        help="Minari expert data level (expert-v0) ",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        required=True,
        help="Minimum reward for episodes",
    )
    parser.add_argument(
        "--intervals",
        type=int,
        default=5,
        help="Number of reward intervals",
    )
    parser.add_argument(
        "--episodes-per-interval",
        type=int,
        default=5,
        help="Number of episodes per interval",
    )

    args = parser.parse_args()
    dataset, env = d3rlpy.datasets.get_minari(
        args.env + "/" + args.expert_level, render_mode="rgb_array"
    )
    # dataset, env = d3rlpy.datasets.get_d4rl("hopper-expert-v0")
    mj_env_name = env.unwrapped.spec.id
    # mj_env = gym.make(mj_env_name, render_mode="rgb_array")

    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    # Get the reward, again
    rewards = np.array([ep.rewards.sum() for ep in dataset.episodes])
    boundaries = np.linspace(args.min_reward, rewards.max(), args.intervals + 1)

    interval_episodes = []  # List of lists. inner list is row for same interval of reward
    for idx in range(args.intervals):
        low = boundaries[idx]
        high = boundaries[idx + 1]

        # Get the episodes in the interval
        if idx < args.intervals - 1:
            sel_idxs = np.where((rewards >= low) & (rewards < high))[0]
        else:
            sel_idxs = np.where((rewards >= low))[0]

        if len(sel_idxs) == 0:
            print(f"No episodes found in interval [{low}, {high}]. Skipping.")
            interval_episodes.append([])
            continue

        chosen = sel_idxs[: args.episodes_per_interval]
        episodes = [dataset.episodes[i] for i in chosen]
        interval_episodes.append(episodes)

    grid_video_path = os.path.join(
        video_dir,
        f"{mj_env_name}_grid.mp4",
    )
    render_episodes_grid_video(
        interval_episodes, env.unwrapped, grid_video_path
    )
    # render_episode(
    #     episodes[0], env.unwrapped, os.path.join(video_dir, "hopper_test.mp4")
    # )
    print(
        f"Saved grid video ({args.intervals} rows x {args.episodes_per_interval} cols) to {grid_video_path}"
    )
