import argparse
import os
from itertools import zip_longest
import xml.etree.ElementTree as ET
import gymnasium as gym
from importlib import resources
import tempfile

import imageio
import numpy as np

import d3rlpy


os.environ["MUJOCO_GL"] = "egl"

def ant_xml():
    base_xml = resources.files("gymnasium.envs.mujoco.assets") / "ant.xml"

    # 2) Parse and recolor the leg subtrees
    tree = ET.parse(base_xml)
    root = tree.getroot()

    PURPLE = "0.60 0.30 0.85 1.0"

    def paint_leg(body_name: str, rgba: str):
        # find the body by name, then recolor all geoms in its subtree
        for body in root.findall(f".//body[@name='{body_name}']"):
            for geom in body.findall(".//geom"):
                geom.set("rgba", rgba)

    LEG_BODIES = ["front_left_leg", "back_leg"]

    for name in LEG_BODIES:
        paint_leg(name, PURPLE)

    # 3) Save to a temp xml and make the env with it (no mujoco import needed)
    with tempfile.NamedTemporaryFile(suffix="_ant_purple.xml", delete=False) as tmp:
        tree.write(tmp.name)
        custom_xml_path = tmp.name
    return custom_xml_path


def rollout_frames_generic(env, episode, missing, max_steps=1000):
    """Run one rollout for generic MuJoCo environments (e.g., Ant, HalfCheetah)."""
    frames = []
    n_qpos = env.unwrapped.model.nq
    n_qvel = env.unwrapped.model.nv
    n_steps = min(episode.actions.shape[0], max_steps)

    obs, info = env.reset()

    n_missing = len(missing)
    qpos = np.zeros(n_qpos)
    qpos[n_missing:] = episode.observations[0, : n_qpos - n_missing]
    qvel = episode.observations[
        0, n_qpos - n_missing : n_qvel - n_missing + n_qpos
    ].copy()
    env.unwrapped.set_state(qpos, qvel)

    print(qpos)
    for i in range(n_steps - 1):
        obs, reward, _, _, info = env.step(episode.actions[i])

        qpos = np.zeros(n_qpos)
        qpos[n_missing:] = episode.observations[i + 1, : n_qpos - n_missing]
        for i, k in enumerate(missing):
            qpos[i] = info.get(k, qpos[i])
        qvel = episode.observations[
            i + 1, n_qpos - n_missing : n_qpos + n_qvel - n_missing
        ]
        env.unwrapped.set_state(qpos, qvel)

        frame = env.render()
        frames.append(frame)

    return frames


def rollout_frames(env, episode, max_steps=1000):
    """Dispatch to environment-specific rollout function."""
    env_id = env.unwrapped.spec.id.lower()
    if "hopper" in env_id or "halfcheetah" in env_id or "walker2d" in env_id:
        return rollout_frames_generic(env, episode, ["x_position"], max_steps)
    elif "ant" in env_id:
        return rollout_frames_generic(
            env, episode, ["x_position", "y_position"], max_steps
        )
    else:
        raise ValueError(
            f"rollout_frames not implemented for environment: {env_id}"
        )


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

    env_name = args.env.replace('/', '_')
    grid_video_path = os.path.join(
        video_dir,
        f"{env_name}_grid.mp4",
    )
    xml = ant_xml()
    mj_env = gym.make("Ant-v5", xml_file=xml, render_mode="rgb_array")
    render_episodes_grid_video(interval_episodes, mj_env, grid_video_path)
    print(
        f"Saved grid video ({args.intervals} rows x {args.episodes_per_interval} cols) to {grid_video_path}"
    )
