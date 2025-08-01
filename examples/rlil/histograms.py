import os

# Enable headless rendering for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

import argparse

import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.signal import windows

import d3rlpy


def sine(t, frequency, phase):
    return np.sin(frequency * t + phase)


def render_episode_video(episode, env, output_path, fps=30, max_frames=None):
    """
    Render a video of an episode from expert demonstrations by stepping the environment with recorded actions.

    Args:
        episode: The expert demonstration episode, containing `episode.actions`.
        env: The MuJoCo environment with `render_mode="rgb_array"`.
        output_path: Path to save the output video (e.g., mp4).
        fps: Frames per second for the generated video.
        max_frames: Optional limit on the number of frames to render.

    Returns:
        The path to the saved video file.
    """
    # Reset the environment and get initial observation
    obs, info = env.reset()

    frames = []
    # Determine how many steps to render
    total_steps = len(episode.actions)
    if max_frames is not None:
        total_steps = min(total_steps, max_frames)

    for i in range(total_steps):
        # Render the current frame
        frame = env.render()
        frames.append(frame)

        # Step the environment with the expert action
        action = episode.actions[i]
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    # Save video using imageio
    print(f"Saving video with {len(frames)} frames to {output_path}")
    imageio.mimsave(output_path, frames, fps=fps)

    return output_path


def get_period(dataset, state_idx):
    n_episodes = len(dataset.episodes)
    periods = np.empty(n_episodes)
    rewards = np.empty(n_episodes)
    start_skip = 200
    for k, episode in enumerate(dataset.episodes):
        # Extract joint positions
        len_episode = episode.observations.shape[0] - start_skip
        if len_episode < 400:
            periods[k] = 0
            continue
        joint_positions = np.empty(len_episode)
        rewards[k] = episode.rewards.sum()
        for i in range(len_episode):
            joint_positions[i] = episode.observations[start_skip + i][state_idx]

        # Remove DC component (mean of the signal)
        joint_positions = joint_positions - np.mean(joint_positions)

        # Set up FFT parameters
        N = len(joint_positions)
        # Assuming sampling rate of 40 Hz based on the original code
        sampling_rate = 125

        # Apply Hann window to minimize spectral leakage
        window = windows.hann(N)
        windowed_signal = joint_positions * window

        # Compute FFT of the windowed signal
        fft_values = fft(windowed_signal)
        # Get positive frequency components (first half of spectrum)
        fft_values = fft_values[: N // 2]
        # Compute power spectrum
        power_spectrum = np.abs(fft_values) ** 2

        # Get frequency axis (positive frequencies only)
        frequencies = fftfreq(N, 1 / sampling_rate)[: N // 2]

        # Exclude the DC component (frequency = 0)
        mask = frequencies > 0
        filtered_frequencies = frequencies[mask]
        filtered_power = power_spectrum[mask]

        # Find the dominant frequency (peak in the power spectrum)
        if len(filtered_frequencies) > 0:
            dominant_freq_idx = np.argmax(filtered_power)
            dominant_freq = filtered_frequencies[dominant_freq_idx]
            # Convert frequency to period
            periods[k] = 1.0 / dominant_freq
        else:
            # Fallback if no valid frequencies found
            periods[k] = 0

    return periods, rewards


def validate_period_plot(
    episode, state_idx, period, sampling_rate=125, start_skip=200, output_path=None
):
    """
    Plot joint state data with fitted sine wave for period validation.
    """
    len_episode = episode.observations.shape[0] - start_skip
    data = episode.observations[start_skip:, state_idx]
    data = data - np.mean(data)
    t = np.arange(len_episode) / sampling_rate
    sine_fit = sine(t, 2 * np.pi / period, 0)
    plt.figure(figsize=(8, 4))
    plt.plot(t, data, label="Data")
    plt.plot(t, sine_fit, label=f"Sine fit (period={period:.2f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel(f"State[{state_idx}]")
    plt.legend()
    plt.title("Validation of extracted period for episode")
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def get_reward(dataset):
    n_episodes = len(dataset.episodes)
    rewards = np.empty(n_episodes)
    for k, episode in enumerate(dataset.episodes):
        len_episode = episode.observations.shape[0]
        rewards[k] = episode.rewards.sum()
    return rewards


def make_hist(periods, fname, title):
    # Generate a histogram of the periods
    plt.figure(figsize=(10, 6))
    plt.hist(periods, bins=40, color="blue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(fname)
    plt.clf()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate reward histogram for Minari expert dataset"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="mujoco/hopper/expert-v0",
        help="Minari environment id",
    )
    args = parser.parse_args()

    dataset, min_env = d3rlpy.datasets.get_minari(args.env)

    # Ensure histogram directory exists
    hist_dir = "histograms"
    os.makedirs(hist_dir, exist_ok=True)

    # Generate histogram of total episode rewards
    rewards = get_reward(dataset)
    env_name = args.env.split("/")[1]
    hist_path = os.path.join(hist_dir, f"{env_name}_reward_hist.png")
    make_hist(rewards, hist_path, f"Reward Histogram for {env_name}")

    print(f"Saved reward histogram for {args.env} to {hist_path}")

    # for idx in selected_indices:
    #     episode = dataset.episodes[idx]
    #     period = torso_periods[idx]
    #     video_path = os.path.join(
    #         selected_video_dir, f"episode_{idx}_torso_period_{period:.2f}.mp4"
    #     )
    #     render_episode_video(
    #         episode, mj_env, video_path, fps=30, max_frames=max_frames_per_video
    #     )
    #     print(f"Generated video for episode {idx} with torso period {period:.2f}")
    #     # Generate validation plot for the period estimation
    #     plot_path = os.path.join(
    #         selected_video_dir,
    #         f"episode_{idx}_torso_period_{period:.2f}_validation.png",
    #     )
    #     validate_period_plot(episode, 0, period, output_path=plot_path)
    #     print(f"Saved validation plot for episode {idx} to {plot_path}")
