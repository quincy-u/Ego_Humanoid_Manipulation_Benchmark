# Isaac-Humanoid-Benchmark
A diverse humanoid benchmark based on Isaac Lab and a testbed for manipulation policy generalization.
## Overview
This is the repo of simulation benchmark for our work:
### EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos

Ruihan Yang<sup>1*</sup>, Qinxi Yu<sup>2*</sup>, Yecheng Wu<sup>3,4</sup>, Rui Yan<sup>1</sup>, Borui Li<sup>1</sup>, An-Chieh Cheng<sup>1</sup>, Xueyan Zou<sup>1</sup>, Yunhao Fang<sup>1</sup>, Xuxin Cheng<sup>1</sup>, Ri-Zhao Qiu<sup>1</sup>, Hongxu Yin<sup>4</sup>, Sifei Liu<sup>4</sup>, Song Han<sup>3,4</sup>, Yao Lu<sup>4</sup>, Xiaolong Wang<sup>1</sup>

<sup>1</sup>UC San Diego / <sup>2</sup>UIUC / <sup>3</sup>MIT / <sup>4</sup>NVIDIA

[Project Page](https://rchalyang.github.io/EgoVLA) / [Arxiv](https://arxiv.org/abs/2507.12440) / [Training code and eval](https://github.com/RchalYang/EgoVLA_Release)

For training code and eval code, follow: https://github.com/RchalYang/EgoVLA_Release

## Getting Started

### Installation
1. **Install IsaacLab**  
   Follow the [IsaacLab Local Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
2. **Clone Ego_Humanoid_Manipulation_Benchmark**  
   ```sh
   git clone https://github.com/quincy-u/Ego_Humanoid_Manipulation_Benchmark.git
   ```

3. **Install the Humanoid Tasks Extension**  
   ```sh
   cd Ego_Humanoid_Manipulation_Benchmark && ${IsaacLab_PATH}/isaaclab.sh -p -m pip install -e source/extensions/humanoid.tasks
   ```

4. **Adjust Rendering and Physics (Recommended)**  
   Replace the following files in your IsaacLab installation with the versions from this repo for adjusted rendering and physics:
   - `source/apps/isaaclab.python.kit`
   - `source/apps/isaaclab.python.rendering.kit`    
   (Do the same for headless apps if needed.)

   Note: In newer versions of IsaacLab, these files are located under `apps/` instead of `source/apps/` and may not be directly replaceable. In that case, update the corresponding settings manually as needed.

---
### Usage

To run an environment:

```sh
${IsaacLab_PATH}/isaaclab.sh -p scripts/<agent_file> --task <environment_name> --num_envs <number_of_environments> --enable_cameras
```

**Example:**  
Run the `Humanoid-Push-Box-v0` environment with random actions and 4 parallel environments:
```sh
${IsaacLab_PATH}/isaaclab.sh -p scripts/random_agent.py --task Humanoid-Push-Box-v0 --num_envs=4 --enable_cameras
```

- `<agent_file>`: Python file specifying the agent (e.g., `random_agent.py`)
- `<environment_name>`: Name of the environment (see table below)

**Env Config:**   
You can override configs directly before `gym.make(args_cli.task, cfg=env_cfg)`.   
Example:
`env_cfg.room_idx = 2
` This changes the background room.
See the full list of configurable parameters below.
## configurable parameters
| #   | Variable Name      | Description                                                                                                   | Type    |
|-----|-------------------|---------------------------------------------------------------------------------------------------------------|---------|
| 1   | `episode_length_s` | Episode length in seconds                                                                                     | float   |
| 2   | `decimation`       | Robot control decimation                                                                                      | int     |
| 3   | `action_scale`     | Action scaling factor                                                                                         | float   |
| 4   | `spawn_table`      | Spawn a table in the environment                                                                              | bool    |
| 5   | `spawn_background` | Spawn a background room in the environment                                                                    | bool    |
| 6   | `room_idx`         | Index of background room to use (1–5)                                                                         | int     |
| 7   | `table_idx`        | Index of table to use (1–3)                                                                                   | int     |
| 8   | `seed`             | Random seed used by Isaac Sim                                                                                 | int     |
| 9   | `randomize`        | Whether to randomize the environment                                                                          | bool    |
| 10  | `randomize_range`  | Randomization range factor                                                                                    | float   |
| 11  | `randomize_idx`    | `< 0`: non-reproducible randomization. Otherwise, reproducible randomization given fixed index                             | int     |

For the details of randomization implementation, check each the `_reset_idx()` implementation of each env file. 
## Available Environments

| #  | Environment Name                   |
|----|------------------------------------|
| 1  | Humanoid-Close-Drawer-v0           |
| 2  | Humanoid-Open-Drawer-v0            |
| 3  | Humanoid-Flip-Mug-v0               |
| 4  | Humanoid-Open-Laptop-v0            |
| 5  | Humanoid-Pour-Balls-v0             |
| 6  | Humanoid-Push-Box-v0               |
| 7  | Humanoid-Stack-Can-v0             |
| 8  | Humanoid-Stack-Can-Into-Drawer-v0 |
| 9  | Humanoid-Unload-Cans-v0            |
| 10 | Humanoid-Insert-Cans-v0            |
| 11 | Humanoid-Sort-Cans-v0              |
| 12 | Humanoid-Insert-And-Unload-Cans-v0 |

---

---
### Notice!
We are using IsaacLab version 1.2.0 and IsaacSim 4.2.0 for EgoVLA data collection and evaluation. As we observe large change of physics behaviours in later IsaacLab & IsaacSim version, please consider downgrade versions to reproduce results.
