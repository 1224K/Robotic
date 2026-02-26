# IsaacLab environment for pick-and-place task

## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

2. Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

3. Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    cd Robotic
    python -m pip install -e source/Robotic
  
4. Download the USD files from Google Drive, unzip them, and place them in the assets directory.
  
    ```bash
    Robotic/
    ├── assets/
    |   ├── Fan.usd
    |   ├── Plate.usd
    |   ├── Rack.usd
    |   └── RobotLeftArm.usd
    ├── scripts/
    │   ├── rl_games/
    │   │   ├── train.py
    │   │   └── play.py
    │   ├── list_envs.py
    │   └── view_env.py                            (Random Agent 測試環境)
    └── source/
        └── Robotic/
            └── Robotic/
                ├── robots/
                │   └── RS_M90E7A.py               (左手臂的配置檔，目前是用速度控制)
                └── tasks/
                    └── direct/
                        └── robotic/
                            ├── robotic_env.py     (如何跟環境互動，獎勵訊號)
                            └── robotic_env_cfg.py (初始要用到的一些參數值)
    
    ```

5. Verify that the extension is correctly installed by:

    - Listing the available tasks:

      Note: It the task name changes, it may be necessary to update the search pattern "Template-"
      (in the `scripts/list_envs.py` file) so that it can be listed.

      ```bash
      # Under the outermost Robotic/ directory
      # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
      python scripts/list_envs.py
      ```

    - Launch a random agent:

      ```bash
      # Under the outermost Robotic/ directory
      # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
      python scripts/view_env.py
      ```

6. Train an RL agent

    ```bash
    # Under the outermost Robotic/ directory
    python scripts/rl_games/train.py --task=Template-Robotic-Direct-v0 \
      agent.params.config.horizon_length=128 \
      agent.params.config.minibatch_size=256 \
      --video \
      --video_length=1000 \
      --num_envs=20 \
      --max_iterations=1000

    # Under the outermost Robotic/ directory
    python scripts/rl_games/train.py --task=Template-Robotic-Direct-v0 \
      agent.params.config.horizon_length=512 \
      agent.params.config.minibatch_size=512 \
      agent.params.config.mini_epochs=6 \
      --num_envs=20 \
      --video \
      --video_length=1000

7. Play with checkpoint

    ```bash
    python scripts/rl_games/play.py --task=Template-Robotic-Direct-v0 --checkpoint=
    ```

## Docs

- Environment Overview

  | Component           | Description                                                         |
  | ------------------- | ------------------------------------------------------------------- |
  | **Robot**           | 7-DOF manipulator (RS_M90E7A) with dual sliders for gripper control |
  | **Objects**         | Rigid bodies loaded via USD files (`Fan`, `Plate`, `Rack`)          |
  | **Scene**           | Includes ground plane, dome light, and physics-based interactions   |
  | **Action Type**     | Continuous velocity control (`Box[-1,1]^7 + gripper [-0.2, 0.2]`)   |
  | **Simulation Step** | `dt = 1/120 s`, with decimation factor of 2                         |
  | **Number of Envs**  | Configurable (`--num_envs` argument)                                |

- Observation Space

  | Feature            | Dimension | Description                                                              |
  | ------------------ | --------- | ------------------------------------------------------------------------ |
  | **EE position**    | 3         | `ee_pos` (env-local / world frame position after subtracting env_origin) |
  | **EE quaternion**  | 4         | `ee_quat` in `(w,x,y,z)`                                                 |
  | **Fan position**   | 3         | `fan_pos` (env-local)                                                    |
  | **Fan quaternion** | 4         | `fan_quat` in `(w,x,y,z)`                                                |
  | **Gripper gap**    | 1         | `abs(Slider10 - Slider9)`                                                |
  | **Prev actions**   | 7         | last action `(Δx,Δy,Δz,Δrx,Δry,Δrz,g)`                                   |
  | **Total**          | **22**    | `3+4+3+4+1+7 = 22`                                                       |

- Action Space

  | Type       | Dimension | Range    | Description                                                      |
  | ---------- | --------- | -------- | ---------------------------------------------------------------- |
  | Continuous | 7         | `[-1,1]` | `Δx,Δy,Δz,Δrx,Δry,Δrz,g` (relative EE command + gripper command) |

- Reward

  | Name       | formula                                 | scale | Description                                                           |
  | ---------- | --------------------------------------- | ----- | -----------------------------------------------------------           |
  | r_reach    |  \|\|ee_pos - fan_pos\|\|               | -1    | Distance error between the end-effector gripper and the fan           |
  | r_degree   | angle_deg                               | -0.5  | Orientation penalty: angle difference between EE and fan (in degrees) |
  | r_grasp    | 𝟙[grasp_confirmed]                      | 200   | Sparse reward when the fan is successfully grasped and held           |
  | r_insert   | \|\|target_pos - fan_pos\|\|            | -1    | Distance error between the target and the fan                         |

- Success Matrix

  | Name       | formula                             | Description                                                 |
  | ---------- | ----------------------------------- | ----------------------------------------------------------- |
  | s_reach    |  \|\|ee_pos - fan_pos\|\| < 0.15    | Whether the end-effector gripper is close enough to the fan |
  | s_grasp    | grasp_confirmed                     | Whether the fan is successfully grasped and held            |
  | s_insert   | \|\|target_pos - fan_pos\|\| < 0.05 | Whether the fan is is close enough to the target            |
