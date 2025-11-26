## Installation

1. Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

2. Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

3. Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    cd Robotic
    python -m pip install -e source/Robotic
  
4. Download the USD files from Google Drive, unzip them, and place them in the project root directory.
  
    ```bash
    Robotic
    - scriptes
    - source
    - Fan.usd
    - Plate.usd
    - Rack.usd
    - RobotLeftArm.usd

5. Verify that the extension is correctly installed by:
  
    - Listing the available tasks:

          Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
          (in the `scripts/list_envs.py` file) so that it can be listed.

          # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
          python scripts/list_envs.py

    - Running a random agent:

          # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
          python scripts/view.py

6. Train an RL agent
    ```bash
    python scripts/rl_games/train.py --task=Template-Robotic-Direct-v0 \
      agent.params.config.horizon_length=128 \
      agent.params.config.minibatch_size=256 \
      --video \
      --num_envs=100

7. Play with checkpoint
    ```bash
    python scripts/rl_games/play.py --task=Template-Robotic-Direct-v0 --checkpoint=