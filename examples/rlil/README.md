# RLIL Visualization Tools

## Overview

1. **Generate reward histogram**: Use `histograms.py` to plot the distribution of total episode rewards and find the reward threshold.
2. **Render grid video**: Use `visualize.py` to create a grid video of rollouts for episodes in different reward intervals, based on the chosen threshold.

## Requirements

Set up Minari as described in the Notion

You also need 
```bash 
$ pip install imageio scipy matplotlib
```

## 1. Generate Reward Histogram

Run the histogram script to visualize the distribution of episode rewards:

```bash
python histograms.py --env <domain>/<task>/<expert-level>
```

- `--env`: Minari environment identifier (e.g., `mujoco/hopper/expert-v0`).

This produces a histogram in the `histograms` folder.

## 2. Render Grid Video of Episodes

Once you've chosen a minimum reward based on the histogram, run the visualization script:

```bash
python visualize.py \
  --env <domain>/<task> \
  --expert-level <expert-level> \
  --min-reward <MIN_REWARD> \
  [--intervals N] \
  [--episodes-per-interval M]
```

Arguments:

- `--env`: Minari environment name without expert suffix (e.g., `mujoco/hopper`).
- `--expert-level`: Expert data level (e.g., `expert-v0` by default).
- `--min-reward`: Minimum reward threshold to include episodes.
- `--intervals`: Number of reward intervals (rows in grid). Default: 5.
- `--episodes-per-interval`: Number of episodes per interval (columns in grid). Default: 5.

Output:

- `videos/<task>_grid.mp4`: Grid video showing rollouts of episodes across reward intervals.

