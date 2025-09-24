import os

# Enable headless rendering for MuJoCo
os.environ["MUJOCO_GL"] = "egl"
import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import d3rlpy


def set_missing(observation, nq, nv, nm, missing=None, info=None):
    qpos = np.zeros(nq)
    qpos[nm:] = observation[: nq - nm]
    if missing is not None and info is not None:
        for i, k in enumerate(missing):
            qpos[i] = info.get(k, qpos[i])

    qvel = observation[nq - nm : nq + nv - nm]
    return qpos, qvel


def replay_episode(env, episode, missing, x):
    """Run one rollout for generic MuJoCo environments (e.g., Ant, HalfCheetah)."""

    n_qpos = env.unwrapped.model.nq
    n_qvel = env.unwrapped.model.nv
    n_steps = episode.actions.shape[0]

    obs, info = env.reset()

    n_missing = len(missing)
    qpos, qvel = set_missing(episode.observations[0], n_qpos, n_qvel, n_missing)
    env.unwrapped.set_state(qpos, qvel)

    exceeds_x = False
    fails = False

    def _correct_angle(thetas):
        return np.where(thetas <= np.pi / 2, thetas, thetas - np.pi)

    def _convert(quat):
        q = quat[:, [0, 3]]
        ms = np.linalg.norm(q, axis=1)
        return 2 * np.arcsin(q[:, 1] / ms)

    c = 200
    yaws = _convert(episode.observations[-c:, 1:5])
    yaw = yaws.mean()
    initial_yaw = _convert(episode.observations[0, 1:5][None])
    turns_optimally = yaw * initial_yaw >= 0

    # print(initial_yaw)
    # a = ""
    # while a != "no":
    #     a = input("give orientation")
    #     w = np.asarray(a.split(","), dtype=float)
    #
    #     qposnew = qpos.copy()
    #     qposnew[3:7] = w
    #     env.unwrapped.set_state(qposnew, qvel)
    #
    #     frame = env.render()
    #
    #     mag = np.sqrt(w[0] ** 2 + w[3] ** 2)
    #     yaw = 2 * np.arccos(w[0] / mag)
    #     print(f"Yaw = {yaw / np.pi:3f} * pi")
    #
    #     plt.imshow(frame)
    #     plt.axis("off")
    #     plt.show()

    for i in range(n_steps - 1):
        obs, reward, terminated, _, info = env.step(episode.actions[i])

        if info["x_position"] > x:
            exceeds_x = True
        if terminated and i < n_steps - 2:
            fails = True

        qpos, qvel = set_missing(
            episode.observations[i + 1],
            n_qpos,
            n_qvel,
            n_missing,
            missing,
            info,
        )
        env.unwrapped.set_state(qpos, qvel)

    return (exceeds_x, fails, turns_optimally, initial_yaw)


def sliding_window(x, y, window):
    smoothed = []
    for xi in x:
        mask = (x >= xi - window / 2) & (x <= xi + window / 2)
        smoothed.append(y[mask].mean() if mask.any() else 0)

    return np.array(smoothed)


def get_reward(dataset):
    n_episodes = len(dataset.episodes)
    rewards = np.empty(n_episodes)
    for k, episode in enumerate(dataset.episodes):
        episode.observations.shape[0]
        rewards[k] = episode.rewards.sum()
    return rewards


def make_plot(xs, ds, fails, turns):
    plt.scatter(xs, sliding_window(xs, ds, 0.1))
    plt.ylabel(f"Reached {100.0}m (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(
        [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
        [
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
        ],
    )
    plt.show()

    plt.scatter(xs, sliding_window(xs, turns, 0.1))
    plt.ylabel("Turned Optimally (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(
        [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
        [
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
        ],
    )
    plt.show()

    plt.scatter(xs, sliding_window(xs, fails, 0.1))
    plt.ylabel("Failure (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(
        [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
        [
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
        ],
    )
    plt.show()

    plt.scatter(
        xs,
        sliding_window(
            xs, np.where(xs >= 0, np.ones_like(xs), np.zeros_like(xs)), 0.1
        ),
    )
    plt.ylabel("Should turn left (density, per rad)")
    plt.xlabel("Initial Orientation (rad)")
    plt.xticks(
        [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2],
        [
            r"$-\frac{\pi}{2}$",
            r"$-\frac{\pi}{4}$",
            r"$0$",
            r"$\frac{\pi}{4}$",
            r"$\frac{\pi}{2}$",
        ],
    )
    plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate behaviour plots")
    parser.add_argument(
        "--env",
        type=str,
        default="mujoco/ant/expert-v0",
        help="Minari environment id",
    )
    args = parser.parse_args()

    dataset, min_env = d3rlpy.datasets.get_minari(
        args.env, render_mode="rgb_array"
    )

    graph_dir = "graphs"
    os.makedirs(graph_dir, exist_ok=True)

    ds = []
    fails = []
    turns = []
    ics = []
    for episode in tqdm(dataset.episodes):
        distance, fail, turn, initial_w = replay_episode(
            min_env, episode, ["x_position", "y_position"], 100.0
        )
        ds.append(distance)
        fails.append(fail)
        turns.append(turn)
        ics.append(initial_w)

    xs = np.array(ics).flatten()
    ds = np.array(ds).flatten()
    turns = np.array(turns).flatten()
    fails = np.array(fails).flatten()

    sort_idx = np.argsort(xs)
    xs = xs[sort_idx]
    ds = ds[sort_idx]
    turns = turns[sort_idx]
    fails = fails[sort_idx]

    print(
        f"Initial distribution {(xs >= 0).sum() / xs.shape[0] * 100:.2f}% should turn left"
    )

    make_plot(xs, ds, fails, turns)

    print("Now filtered by exceeding distance")
    xs2 = xs[ds]
    ds2 = ds[ds]
    fails2 = fails[ds]
    turns2 = turns[ds]
    make_plot(xs2, ds2, fails2, turns2)

    print("Now filtered by not failing")

    xs3 = xs[~fails]
    ds3 = ds[~fails]
    fails3 = fails[~fails]
    turns3 = turns[~fails]
    make_plot(xs3, ds3, fails3, turns3)
