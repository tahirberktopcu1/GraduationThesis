# RL-Based Mobile Robot Path Planning - Detailed Report

## 1) Project Summary
This study evaluates a PPO-based reinforcement learning agent navigating to a goal in a fixed 3D scene (movement on the XZ plane) and compares it against A* on the same map.

## 2) Experimental Setup
- Environment: Single fixed map, static obstacles.
- Motion: Continuous actions (forward/backward and turn).
- Observations: goal direction + normalized distance + velocity + raycast sensor values.
- Reward: progress toward goal, per-step penalty, extra reward on goal reach.

## 3) Evaluation Methodology
- The RL model was run in inference mode for 64 episodes (provided subset).
- Success, collision, and timeout rates; average steps and path length were computed.
- A* was run on the same map to obtain a reference path length.

## 4) Quantitative Results (RL)
- Episodes: 64
- Success rate: 1.00 (100.00%)
- Collision rate: 0.00 (0.00%)
- Timeout rate: 0.00 (0.00%)
- Average steps: 479.64 (std 2.61)
- Average path length: 50.62 (std 1.44)
- Average reward: 5.64 (std 1.44)
- Mean step length (path/steps): 0.1055

## 5) A* Result
- A* path length: 28.97
- Node count: 49
- Mean grid step length (path/(nodes-1)): 0.604

## 6) RL vs A* Comparison
- Path length difference (RL - A*): 21.65
- Path length ratio (RL/A*): 1.75
- In this subset, RL succeeds in all episodes, while A* yields shorter paths.
- A* is grid-based with full map knowledge, so it is expected to be more optimal.

## 7) Figures
- steps_vs_episode.png
- path_length_vs_episode.png
- reward_vs_episode.png
- rl_vs_astar_path_length.png
- path_length_histogram.png
- path_length_ratio_vs_episode.png
- steps_vs_path_length_scatter.png

## 8) Discussion
- The RL policy shows stable continuous control, but path optimality is worse than A*.
- A* is deterministic and produces shorter paths on a fixed map, but it must replan for new maps.
- RL can generalize to similar maps with the same policy, trading off optimality for adaptability.

## 9) Limitations
- Single fixed map and static obstacles only.
- No real robot or real sensor noise; purely simulation-based.
- Limited hyperparameter tuning.

## 10) Deliverables
- Metrics summaries, figures, and this report in this folder.
- nav_metrics_subset.csv (64-episode evaluation data).
- comparison_table.csv (RL vs A* comparison).
