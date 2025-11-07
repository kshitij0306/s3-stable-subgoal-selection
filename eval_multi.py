import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TypeVar

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import s3.s3 as s3_algo
from envs import EnvWithGoal, GatherEnv
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env


def make_env(env_name: str, seed: int):
    if env_name == "AntGather":
        env = GatherEnv(create_gather_env(env_name, seed), env_name)
    elif env_name in ["AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(env_name, seed), env_name)
    else:
        raise NotImplementedError(f"Unsupported env {env_name}")
    env.seed(seed)
    return env


def init_agents(env, args, algo: str, seed: int, model_dir: str):
    low = np.array((-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3))
    high = -low
    man_scale = (high - low) / 2
    if args.env_name == "AntFall":
        controller_goal_dim = 3
    else:
        controller_goal_dim = 2
    if args.absolute_goal:
        man_scale[0] = 30
        man_scale[1] = 30
        no_xy = False
    else:
        no_xy = True

    obs = env.reset()
    goal = obs["desired_goal"]
    state = obs["observation"]

    max_action = float(env.action_space.high[0])
    action_dim = env.action_space.shape[0]

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    if args.env_name in ["AntMaze", "AntPush", "AntFall"]:
        goal_dim = goal.shape[0]
    else:
        goal_dim = 0

    controller_policy = s3_algo.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=0,
        critic_lr=0,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=0,
        noise_clip=0
    )

    manager_policy = s3_algo.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=0,
        critic_lr=0,
        candidate_goals=10,
        correction=True,
        scale=man_scale,
        goal_loss_coeff=10.,
        absolute_goal=args.absolute_goal
    )

    suffix = None
    if args.checkpoints:
        suffix = args.checkpoints.get(algo)

    if args.load:
        base_dir = Path(model_dir)
        try:
            manager_policy.load(base_dir, args.env_name, algo, suffix=suffix)
            controller_policy.load(base_dir, args.env_name, algo, suffix=suffix)
            tag = suffix or "base"
            print(f"[{algo}] Loaded checkpoint '{tag}' from {base_dir}.")
        except Exception as exc:
            print(f"[{algo}] Failed to load weights from {base_dir}: {exc}")

    return manager_policy, controller_policy, controller_goal_dim, device


def collect_subgoal_scatter(env,
                            manager_policy,
                            controller_policy,
                            controller_goal_dim: int,
                            freq: int,
                            eval_episodes: int,
                            absolute_goal: bool) -> np.ndarray:
    samples: List[Tuple[float, float]] = []

    with torch.no_grad():
        for _ in range(eval_episodes):
            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]

            done = False
            step_count = 0
            active = None
            subgoal = None

            while not done:
                if step_count % freq == 0:
                    subgoal = manager_policy.sample_goal(state, goal)
                    start_xy = state[:controller_goal_dim].copy()
                    sg_target = subgoal[:controller_goal_dim].copy()
                    if not absolute_goal:
                        sg_target = start_xy + sg_target
                    active = {
                        "start_xy": start_xy,
                        "target_xy": sg_target,
                        "steps_taken": 0,
                    }

                action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, _ = env.step(action)
                new_state = new_obs["observation"]
                goal = new_obs["desired_goal"]

                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)
                step_count += 1

                if active is not None:
                    active["steps_taken"] += 1
                    finished_window = active["steps_taken"] >= freq or done
                    if finished_window:
                        landing_xy = new_state[:controller_goal_dim].copy()
                        init_dist = float(np.linalg.norm(active["target_xy"] - active["start_xy"]))
                        final_dist = float(np.linalg.norm(active["target_xy"] - landing_xy))
                        samples.append((init_dist, final_dist))
                        active = None

                state = new_state

            if active is not None:
                landing_xy = state[:controller_goal_dim].copy()
                init_dist = float(np.linalg.norm(active["target_xy"] - active["start_xy"]))
                final_dist = float(np.linalg.norm(active["target_xy"] - landing_xy))
                samples.append((init_dist, final_dist))

    if not samples:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(samples, dtype=np.float32)


def collect_trajectories(env,
                         manager_policy,
                         controller_policy,
                         controller_goal_dim: int,
                         freq: int,
                         traj_episodes: int,
                         absolute_goal: bool) -> List[np.ndarray]:
    """
    Collect XY trajectories (in env/world coordinates) for 'traj_episodes' successful episodes.
    Returns: list of arrays with shape (T_i, 2), one per episode.
    """
    trajectories: List[np.ndarray] = []
    with torch.no_grad():
        while len(trajectories) < traj_episodes:
            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]
            done = False
            step_count = 0
            subgoal = None
            xy = [state[:controller_goal_dim].copy()]

            while not done:
                if step_count % freq == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, info = env.step(action)
                new_state = new_obs["observation"]
                goal = new_obs["desired_goal"]
                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)
                state = new_state
                step_count += 1
                xy.append(state[:controller_goal_dim].copy())

            # Save trajectory regardless of success; if you prefer success-only, gate on info["is_success"] if available
            trajectories.append(np.asarray(xy, dtype=np.float32))
    return trajectories


def plot_scatter(scatter_map: Dict[str, np.ndarray],
                 freq: int,
                 env_name: str,
                 output_dir: Path,
                 filename: str,
                 labels: Dict[str, str]):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 7.5))
    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]
    scatter_items = list(scatter_map.items())

    plot_entries = []
    for idx, (algo, data) in enumerate(scatter_items):
        if data.size == 0:
            continue
        plot_entries.append({
            "data": data,
            "color": colors[idx % len(colors)],
            "label": labels.get(algo, algo.upper()),
        })

    if not plot_entries:
        return

    for entry in plot_entries[1:]:
        plt.scatter(entry["data"][:, 0], entry["data"][:, 1], s=26, alpha=0.65,
                    label=entry["label"], color=entry["color"])

    primary_entry = plot_entries[0]
    plt.scatter(primary_entry["data"][:, 0], primary_entry["data"][:, 1], s=26, alpha=0.65,
                label=primary_entry["label"], color=primary_entry["color"])

    max_lim = 0.0
    for data in scatter_map.values():
        if data.size:
            max_val = float(data.max())
            if max_val > max_lim:
                max_lim = max_val
    max_lim = max(1.0, max_lim * 1.05)
    # max_lim = 4.5

    plt.plot([0, max_lim], [0, max_lim], linestyle="--", color="k", linewidth=1.2, label="y = x")
    plt.xlim(0, max_lim)
    plt.ylim(0, max_lim)
    plt.xlabel(r"Current state $s_t$ to subgoal $g_t$ distance")
    plt.ylabel(r"Landing state $s_{t+c}$ to subgoal $g_t$ distance")
    # plt.title(rf"{env_name}: Manager interval c={freq}")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    img_path_png = output_dir / f"{filename}.png"
    img_path_svg = output_dir / f"{filename}.svg"
    data_path = output_dir / f"{filename}.npz"
    plt.savefig(img_path_png, dpi=200)
    plt.savefig(img_path_svg, dpi=200)
    plt.close()

    np.savez_compressed(
        data_path,
        **{algo: data for algo, data in scatter_map.items()},
        manager_interval=freq
    )

    print(f"[PLOT] Saved scatter: {img_path_png}")
    print(f"[PLOT] Saved scatter: {img_path_svg}")
    print(f"[DATA] Saved samples: {data_path}")


def plot_trajectories(traj_map: Dict[str, List[np.ndarray]],
                      env_name: str,
                      output_dir: Path,
                      filename: str,
                      labels: Dict[str, str],
                      maze_bg: Optional[Path] = None,
                      world_bounds: Optional[Tuple[float, float, float, float]] = None,
                      traj_alpha: float = 0.85,
                      traj_lw: float = 2.0):
    """
    Plot XY trajectories as lines; optionally overlay a maze image aligned to world coordinates.
    - maze_bg: path to background image (top-down). If provided, you *should* pass world_bounds=(xmin,xmax,ymin,ymax)
               so the image aligns with the environment coordinates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 7.5))

    # Optional maze background
    bounds = world_bounds if world_bounds is not None else (-2.0, 10.0, -2.0, 10.0)
    xmin, xmax, ymin, ymax = bounds

    if maze_bg is not None and maze_bg.exists():
        img = plt.imread(str(maze_bg))
        if world_bounds is None:
            # Fallback: fill canvas; not guaranteed aligned
            plt.imshow(img, origin="lower", alpha=0.9)
        else:
            plt.imshow(img, extent=[xmin, xmax, ymin, ymax], origin="lower", alpha=0.9)
    else:
        # Default Ant Maze outline if no background image is supplied
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin],
                 linewidth=2, zorder=1, color="black")
        # Middle wall for Ant maze layout
        plt.gca().add_patch(patches.Rectangle((-2, 2), 8, 4,
                                              facecolor="gray",
                                              edgecolor="k",
                                              alpha=0.35,
                                              zorder=0))

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]
    for idx, (algo, trajs) in enumerate(traj_map.items()):
        color = colors[idx % len(colors)]
        label = labels.get(algo, algo.upper())
        # draw each episode
        n_traj = len(trajs)
        if n_traj == 0:
            continue
        if n_traj == 1:
            alpha_values = [traj_alpha]
        else:
            low_alpha = max(0.15, traj_alpha * 0.4)
            alpha_values = np.linspace(traj_alpha, low_alpha, n_traj)
        for t_i, (xy, alpha_val) in enumerate(zip(trajs, alpha_values)):
            # place label only on the first line per algo for a clean legend
            kw = dict(label=label) if t_i == 0 else {}
            plt.plot(xy[:, 0], xy[:, 1], color=color, alpha=float(alpha_val),
                     linewidth=traj_lw, zorder=2, **kw)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("x (world)")
    plt.ylabel("y (world)")
    plt.title(f"{env_name}: Trajectory overlays")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.tight_layout()

    img_path_png = output_dir / f"{filename}_traj.png"
    img_path_svg = output_dir / f"{filename}_traj.svg"
    plt.savefig(img_path_png, dpi=200)
    plt.savefig(img_path_svg, dpi=200)
    plt.close()
    print(f"[PLOT] Saved trajectories: {img_path_png}")
    print(f"[PLOT] Saved trajectories: {img_path_svg}")


def parse_checkpoint_map(args) -> Dict[str, str]:
    return broadcast_arg_to_algos(
        values=args.checkpoint,
        algos=args.algos,
        arg_name="--checkpoint"
    )


def parse_model_dir_map(args) -> Dict[str, str]:
    """
    Returns a dict algo -> model_dir using --model_dirs if provided.
    - If omitted, returns {} and the caller should fall back to --model_dir.
    - If one value is provided, it is broadcast to all algos.
    - If N values are provided, they must match the length/order of --algos.
    """
    return broadcast_arg_to_algos(
        values=args.model_dirs,
        algos=args.algos,
        arg_name="--model_dirs"
    )

def broadcast_arg_to_algos(values: Optional[List[str]],
                           algos: List[str],
                           arg_name: str) -> Dict[str, str]:
    """
    Helper that maps optional CLI values onto each algorithm, supporting:
    - no value (returns {})
    - single value broadcast to all algos
    - per-algo values (length must match --algos)
    """
    if values is None:
        return {}
    if len(values) == 1:
        return {algo: values[0] for algo in algos}
    if len(values) != len(algos):
        raise ValueError(
            f"{arg_name} expects either a single entry or one per algorithm (got {len(values)} values for {len(algos)} algos)."
        )
    return {algo: value for algo, value in zip(algos, values)}


T = TypeVar("T")


def broadcast_list(values: Optional[List[T]],
                   target_len: int,
                   arg_name: str,
                   default: Optional[T] = None) -> List[T]:
    """
    Broadcasts a list of CLI arguments to the expected length.
    - If 'values' is None, fills with the provided default (required in that case).
    - If 'values' has one item, repeats it to match target_len.
    - If 'values' length equals target_len, returns as-is.
    """
    if values is None:
        if default is None:
            raise ValueError(f"{arg_name} requires a value when no default is provided.")
        return [default] * target_len
    if len(values) == 1:
        return [values[0]] * target_len
    if len(values) != target_len:
        raise ValueError(
            f"{arg_name} expects either a single value or {target_len} values (got {len(values)})."
        )
    return values


def main():
    parser = argparse.ArgumentParser(description="Compare one or more algorithms via subgoal scatter plots + trajectory overlays.")
    parser.add_argument("--algos", nargs="+", required=True,
                        help="Names of one or more algorithms to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Seeds for each algorithm; provide one to broadcast or one per algo.")
    parser.add_argument("--seed", type=int, default=0, help="Fallback seed if --seeds not provided.")
    parser.add_argument("--gid", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--env_name", type=str, default="AntMazeSparse")
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--manager_propose_freq", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="./pretrained_models",
                        help="Default model directory used if --model_dirs not provided.")
    parser.add_argument("--model_dirs", nargs="+",
                        help="Per-algo model directories (1 path broadcast or 2 paths in --algos order).")
    parser.add_argument("--plots_dir", type=str, default="./plots")
    parser.add_argument("--plot_name", type=str, default=None)
    parser.add_argument("--checkpoint", nargs="+", help="Checkpoint suffix(es); 1 value broadcast or per-algo.")
    parser.add_argument("--load", action="store_true", help="Load pretrained weights from model_dir(s).")
    parser.add_argument("--absolute_goal", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true", help="Kept for parity; not used here.")
    parser.add_argument("--labels", nargs="+", help="Legend labels for the algorithms (order matches --algos).")

    # New: trajectory plotting options
    parser.add_argument("--traj_episodes", type=int, default=10, help="Number of episodes per algo to plot as trajectories.")
    parser.add_argument("--maze_bg", type=str, default=None, help="Path to a top-down maze image to use as background.")
    parser.add_argument("--world_bounds", nargs=4, type=float, default=None,
                        help="World extents [xmin xmax ymin ymax] to align the background image.")
    parser.add_argument("--traj_alpha", type=float, default=0.85, help="Trajectory line alpha.")
    parser.add_argument("--traj_lw", type=float, default=2.0, help="Trajectory line width.")

    args = parser.parse_args()
    args.checkpoints = parse_checkpoint_map(args)
    args.model_dirs_map = parse_model_dir_map(args)

    labels_map = broadcast_arg_to_algos(args.labels, args.algos, "--labels")
    seeds = broadcast_list(args.seeds, len(args.algos), "--seeds", default=args.seed)
    scatter_map: Dict[str, np.ndarray] = {}
    traj_map: Dict[str, List[np.ndarray]] = {}

    for algo, seed in zip(args.algos, seeds):
        model_dir = args.model_dirs_map.get(algo, args.model_dir)
        print(f"\n=== Evaluating {algo} (seed {seed}) ===")
        print(f"[{algo}] Using model_dir: {model_dir}")
        env = make_env(args.env_name, seed)
        manager_policy, controller_policy, controller_goal_dim, device = init_agents(
            env, args, algo, seed, model_dir=model_dir
        )

        # Collect scatter samples
        samples = collect_subgoal_scatter(
            env=env,
            manager_policy=manager_policy,
            controller_policy=controller_policy,
            controller_goal_dim=controller_goal_dim,
            freq=args.manager_propose_freq,
            eval_episodes=args.eval_episodes,
            absolute_goal=args.absolute_goal,
        )
        scatter_map[algo] = samples

        # Collect trajectories
        trajs = collect_trajectories(
            env=env,
            manager_policy=manager_policy,
            controller_policy=controller_policy,
            controller_goal_dim=controller_goal_dim,
            freq=args.manager_propose_freq,
            traj_episodes=args.traj_episodes,
            absolute_goal=args.absolute_goal,
        )
        traj_map[algo] = trajs

        # Close env
        try:
            env.close()
        except Exception:
            pass
        torch.cuda.empty_cache()
        print(f"[{algo}] Collected {samples.shape[0]} subgoal samples; {len(trajs)} trajectories.")

    plot_base = args.plot_name
    if not plot_base:
        if len(args.algos) == 1:
            plot_base = f"{args.env_name}_{args.algos[0]}"
        else:
            plot_base = f"{args.env_name}_{'_vs_'.join(args.algos)}"

    plots_dir = Path(args.plots_dir)

    # Scatter (unchanged)
    plot_scatter(
        scatter_map=scatter_map,
        freq=args.manager_propose_freq,
        env_name=args.env_name,
        output_dir=plots_dir,
        filename=f"{plot_base}_scatter",
        labels=labels_map,
    )

    # Trajectory overlays (new)
    maze_bg_path = Path(args.maze_bg) if args.maze_bg else None
    world_bounds = tuple(args.world_bounds) if args.world_bounds is not None else None
    plot_trajectories(
        traj_map=traj_map,
        env_name=args.env_name,
        output_dir=plots_dir,
        filename=f"{plot_base}",
        labels=labels_map,
        maze_bg=maze_bg_path,
        world_bounds=world_bounds,
        traj_alpha=args.traj_alpha,
        traj_lw=args.traj_lw,
    )


if __name__ == "__main__":
    main()
