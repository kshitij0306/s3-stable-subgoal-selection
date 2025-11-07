# S3: Stable Subgoal Selection

S3 (Stable Subgoal Selection by Constraining Uncertainty of Coarse Dynamics) is a hierarchical reinforcement learning agent for sparse-reward MuJoCo control tasks such as AntMaze, AntPush, and AntGather. This repository contains the full training and evaluation stack for the continuous-control experiments.

---

## Repository Tour

- `main.py` – entry point for training the continuous-control (MuJoCo) agent.
- `eval.py` / `eval_multi.py` – single-policy evaluation and multi-policy visualization (scatter + trajectory plots).
- `s3/` – agent implementation, models, replay buffers, reachability and dispersion utilities.
- `envs/` – Ant-based maze environments and Gather task support.
- `req.txt` – pinned Python dependencies.
- `cuda_test.py` – quick script to confirm that PyTorch sees your GPU.

---

## Requirements (with versions)

### System
- 64-bit Linux (Ubuntu 20.04/22.04) or WSL2 is recommended for `mujoco_py`. macOS works for CPU evaluation but MuJoCo support is limited.
- NVIDIA GPU with CUDA 11.7 and driver ≥ 515 for fastest training (CPU is supported but significantly slower).
- MuJoCo 2.1.0 binaries extracted to `~/.mujoco/mujoco210`.
- C/C++ toolchain and MuJoCo runtime dependencies: `build-essential`, `libosmesa6-dev`, `libgl1-mesa-dev`, `libglew-dev`, `patchelf`.

### Python
- Python 3.9 or 3.10 (64-bit).
- `pip` ≥ 23.0.1 or `uv`/`poetry` equivalent.

### Key Python Packages (installed via `pip install -r req.txt`)

| Package | Version / Constraint | Purpose |
| --- | --- | --- |
| `torch` | 1.13.1 (CUDA 11.7 build) | Core deep learning |
| `torchvision` | 0.14.1 | Torch vision ops needed by some utilities |
| `torchaudio` | 0.13.1 | Version-locked with PyTorch |
| `gym` | 0.20.0 (OpenAI archive Git pin) | Legacy MuJoCo envs (Ant* tasks) |
| `mujoco_py` | 2.1.2.14 | Python bindings for MuJoCo 2.1 |
| `cython` | 0.29.37 | Required to build `mujoco_py` |
| `numpy` | ≥ 1.18 | Numerical ops |
| `scipy`, `pandas`, `matplotlib` | latest | Analysis/visualization |
| `tensorboard`, `tqdm` | latest | Logging and progress |

> **Note:** `req.txt` already encodes these pins. If you need CPU-only PyTorch, install the matching CPU wheels before running `pip install -r req.txt`.

---

## Environment Setup

1. **Clone and enter the repo**
   ```bash
   git clone <this_repo> hrac-s3
   cd hrac-s3
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   ```

3. **Install MuJoCo 2.1**
   ```bash
   mkdir -p ~/.mujoco && cd ~/.mujoco
   curl -LO https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
   tar -xzf mujoco210-linux-x86_64.tar.gz
   export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
   export MUJOCO_GL=egl  # use osmesa if running purely headless
   cd -
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r req.txt
   ```
   If you see build failures inside `mujoco_py`, double-check that the system packages listed in the requirements section are installed and that `GL`, `GLEW`, and `patchelf` are available.

5. **(Optional) Confirm GPU visibility**
   ```bash
   python cuda_test.py
   ```
   The last line should print your CUDA device name. If `torch.cuda.is_available()` is `False`, reinstall the matching CUDA-enabled PyTorch wheel or run on CPU.

---

## Running S3 (continuous-control tasks)

### Quick Start
```bash
# Train Ant Maze with checkpoints + TensorBoard logging
python main.py \
  --env_name AntMaze \
  --seed 0 \
  --gid 0 \
  --max_timesteps 5e6 \
  --eval_freq 20000 \
  --model_dir models/antmaze_seed0 \
  --save_models \
  --save_halfway_checkpoint
```

Key flags:
- `--env_name`: one of `AntMaze`, `AntMazeSparse`, `AntPush`, `AntFall`, `AntGather`.
- `--gid`: CUDA device index.
- `--save_models`, `--save_halfway_checkpoint`, `--save_periodic`: opt-in checkpointing under `--model_dir`.
- `--binary_int_reward`, `--disable_adj_net`: toggle ablations.

| Environment | Description | Command (minimal) |
| --- | --- | --- |
| Ant Maze (dense) | 2D navigation with shaped rewards | `python main.py --env_name AntMaze` |
| Ant Maze Sparse | Sparse goal signal | `python main.py --env_name AntMazeSparse --binary_int_reward` |
| Ant Push / Fall | Manipulation variants | `python main.py --env_name AntPush` / `--env_name AntFall` |
| Ant Gather | Collect apples in arena | `python main.py --env_name AntGather --inner_dones` (set automatically) |

Each run produces:
- `logs/<algo>/events.out.tfevents.*` – Scalar summaries viewable in TensorBoard.
- `models/<env>_<algo>_*` – Manager/Controller checkpoints (`*_actor.pth`, `*_critic.pth`, etc.).
- `results/` – JSON summaries produced on evaluation intervals.

### Evaluation
```bash
# Evaluate a saved policy from your checkpoint directory
python eval.py \
  --env_name AntMazeSparse \
  --model_dir models/antmaze_seed0 \
  --checkpoint base \
  --eval_episodes 100 \
  --render  # add --video --video_dir videos to export mp4s
```

- Use `--checkpoint half` / custom suffix to load halfway checkpoints created with `--save_halfway_checkpoint`.
- `--sample_n <N>` + `--plots_dir` saves landing scatter plots instead of full evaluations.
- `--heatmap` overlays a success-density map (AntMaze variants).

### Comparing multiple algorithms / visual diagnostics
```bash
python eval_multi.py \
  --algos s3 baseline \
  --env_name AntMazeSparse \
  --model_dirs models/s3_run models/baseline_run \
  --load \
  --plots_dir plots/antmaze_sparse \
  --plot_name scatter_vs \
  --eval_episodes 50 \
  --traj_episodes 10 \
  --manager_propose_freq 10
```
This script builds subgoal dispersion scatter plots and XY trajectory overlays for each algorithm into `plots/`.

