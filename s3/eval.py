import torch

import numpy as np
import os

import s3.s3 as s3_algo
from envs import EnvWithGoal, GatherEnv
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env
from gym.wrappers import RecordVideo
from envs.gymnasium_goal_env import GymnasiumGoalEnv, GYMNASIUM_ENV_SPECS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def collect_landings(env, controller_policy, controller_goal_dim, st, gt, N, c):
    """
    Runs N rollouts from a single (s_t, g_t). Returns:
      samples_xy: (N,2) landings,
      s0: the start xy observed on the first rollout,
      g0: the goal xy used (first 2 dims of gt)
    """
    samples_xy = []
    s0 = None
    g0 = np.asarray(gt[:controller_goal_dim], dtype=np.float32) if gt is not None else None

    for _ in range(int(N)):
        obs0 = env.reset(st)
        state = obs0["observation"].copy()
        pos = obs0["achieved_goal"].copy()   # xy
        if s0 is None:
            s0 = obs0["achieved_goal"].copy()

        # relative subgoal for controller
        if g0 is not None:
            subgoal = g0 - np.asarray(s0[:controller_goal_dim], dtype=np.float32)
        else:
            raise ValueError("collect_landings requires a gt vector")

        for _ in range(int(c)):
            action = controller_policy.select_action(state, subgoal, evaluation=False)
            new_obs, _, done, _info = env.step(action)
            new_state = new_obs["observation"]
            new_pos = new_obs["achieved_goal"]
            subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)
            state = new_state
            pos = new_pos
            if done:
                break

        samples_xy.append(pos.copy())

    return np.asarray(samples_xy), s0, g0
def sample_and_plot_landings(env, args, manager_policy, controller_policy,
                             controller_goal_dim, plots_dir):
    """Run N rollouts from (s_t,g_t), save samples and aligned scatter."""
    N = int(args.sample_n)
    c = int(args.manager_propose_freq)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    samples_xy, s0, g0 = collect_landings(
        env=env,
        controller_policy=controller_policy,
        controller_goal_dim=controller_goal_dim,
        st=args.st,
        gt=args.gt,
        N=N,
        c=c
    )

    npy_path = Path(plots_dir) / f"{args.env_name}_{args.algo}_stc_samples_c{c}_N{N}.npy"
    np.save(npy_path, samples_xy)
    print(f"[DATA] Saved: {npy_path}")

    xmin, xmax, ymin, ymax = -2, 10, -2, 10

    plt.figure(figsize=(7.5, 7.5))
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], linewidth=2, zorder=1)
    plt.gca().add_patch(patches.Rectangle((-2,2), 8, 4, facecolor="gray", edgecolor="k", alpha=0.35, zorder=0))

    plt.scatter([s0[0]], [s0[1]], s=90, marker="s", label="$s_t$", zorder=2)
    if samples_xy.size:
        plt.scatter(samples_xy[:, 0], samples_xy[:, 1], alpha=0.55, s=16, label=r"$s_{t+c}$ landings", zorder=2)
    if g0 is not None and len(g0) >= 2:
        plt.scatter([g0[0]], [g0[1]], s=110, marker="*", label="$g_t$", zorder=3)
    plt.xlim(xmin, xmax); plt.ylim(ymin, ymax); plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.title(rf"P($s_{{t+c}}|s_t,g_t$) landings  (c={c}, N={N})  [{args.algo.upper()}]")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend(loc="upper left"); plt.tight_layout()
    out_path = Path(plots_dir) / f"{args.env_name}_{args.algo}_stc_cloud_c{c}_N{N}.png"
    plt.savefig(out_path, dpi=180)
    print(f"[PLOT] Saved: {out_path}")

def generate_variance_heatmap(env, args, manager_policy, controller_policy,
                              controller_goal_dim, plots_dir):
    """
    Auto-discretizes the XY workspace, repeatedly samples landings from many (s_t,g_t) pairs,
    and plots a heatmap of total variance tr(Cov) at each grid cell.
    - We place a small fixed offset goal from each start: gt = st + (dx, dy).
    - Heatmap cell value = trace(cov(samples)) from that start cell.
    """
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # plotting bounds must match your scatter figure
    xmin, xmax, ymin, ymax = -2, 10, -2, 10
    c = int(args.manager_propose_freq)

    # Grid resolution and sampling budget
    bins = getattr(args, "heatmap_bins", 20)  # adjustable
    # keep per-cell samples modest unless you override sample_n
    per_cell = max(5, int(getattr(args, "sample_n", 50)) // 5)

    # fixed local offset to probe reliability; tweak if desired
    dx, dy = 1.0, 1.0

    xs = np.linspace(xmin + 0.1, xmax - 0.1, bins)
    ys = np.linspace(ymin + 0.1, ymax - 0.1, bins)

    H = np.full((bins, bins), np.nan, dtype=float)  # variance map

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            st = np.array([x, y], dtype=np.float32)
            gt = np.array([x + dx, y + dy], dtype=np.float32)

            try:
                samples_xy, s0, g0 = collect_landings(
                    env=env,
                    controller_policy=controller_policy,
                    controller_goal_dim=controller_goal_dim,
                    st=st,
                    gt=gt,
                    N=per_cell,
                    c=c
                )
            except Exception as e:
                # skip cells where reset or rollout is invalid
                # print(f"[WARN] cell ({i},{j}) failed: {e}")
                continue

            if samples_xy.shape[0] >= 2:
                cov = np.cov(samples_xy, rowvar=False)  # 2x2
                H[j, i] = float(np.trace(cov))          # Var(x)+Var(y)

    # --- Plot overlay ---
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], linewidth=2, zorder=1)
    ax.add_patch(patches.Rectangle((-2,2), 8, 4, facecolor="gray", edgecolor="k", alpha=0.35, zorder=0))

    im = ax.imshow(H, origin='lower',
                   extent=[xmin, xmax, ymin, ymax], aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Variance (trace of Cov): Var(x)+Var(y)")

    ax.set_title(rf"Variance heatmap of $P(s_{{t+c}}|s_t,g_t)$  (c={c}, per-cell N={per_cell})  [{args.algo.upper()}]")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(False)
    fig.tight_layout()

    out_png = Path(plots_dir) / f"{args.env_name}_{args.algo}_stc_heatmap_c{c}_bins{bins}_n{per_cell}.png"
    out_npy = Path(plots_dir) / f"{args.env_name}_{args.algo}_stc_heatmap.npy"
    fig.savefig(out_png, dpi=180)
    np.save(out_npy, dict(H=H, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, bins=bins))
    print(f"[PLOT] Saved heatmap: {out_png}")
    print(f"[DATA] Saved grid:   {out_npy}")

def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_episodes=100, render=False):
    print("Starting evaluation...")
    env.evaluate = True

    with torch.no_grad():
        avg_reward = 0.
        avg_controller_rew = 0.
        global_steps = 0
        goals_achieved = 0
        for eval_ep in range(eval_episodes):
            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]

            done = False
            step_count = 0
            env_goals_achieved = 0
            while not done:
                if step_count % manager_propose_frequency == 0:
                    subgoal = manager_policy.sample_goal(state, goal)

                step_count += 1
                global_steps += 1
                action = controller_policy.select_action(state, subgoal, evaluation=True)
                new_obs, reward, done, _ = env.step(action)
                if env_name != "AntGather" and env.success_fn(reward):
                    env_goals_achieved += 1
                    goals_achieved += 1
                    done = True

                goal = new_obs["desired_goal"]
                new_state = new_obs["observation"]

                subgoal = controller_policy.subgoal_transition(state, subgoal, new_state)

                avg_reward += reward
                avg_controller_rew += calculate_controller_reward(state, subgoal, new_state, ctrl_rew_scale)

                state = new_state

        avg_reward /= eval_episodes
        avg_controller_rew /= global_steps
        avg_step_count = global_steps / eval_episodes
        avg_env_finish = goals_achieved / eval_episodes

        print("---------------------------------------")
        print("Evaluation over {} episodes:\nAvg Ctrl Reward: {:.3f}".format(eval_episodes, avg_controller_rew))
        if env_name == "AntGather":
            print("Avg reward: {:.1f}".format(avg_reward))
        else:
            print("Goals achieved: {:.1f}%".format(100*avg_env_finish))
        print("Avg Steps to finish: {:.1f}".format(avg_step_count))
        print("---------------------------------------")

        env.evaluate = False
        return avg_reward, avg_controller_rew, avg_step_count, avg_env_finish


def get_reward_function(dims, absolute_goal=False, binary_reward=False):
    if absolute_goal and binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(subgoal - next_z, axis=-1) <= 1) * scale
            return reward
    elif absolute_goal:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(subgoal - next_z, axis=-1) * scale
            return reward
    elif binary_reward:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = float(np.linalg.norm(z + subgoal - next_z, axis=-1) <= 1) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def eval_s3(args):
    gym_render_mode = None
    if args.video:
        gym_render_mode = "rgb_array"
    elif args.render:
        gym_render_mode = "human"

    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    elif args.env_name in ["AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    elif args.env_name in GYMNASIUM_ENV_SPECS:
        env = GymnasiumGoalEnv(args.env_name, seed=args.seed, render_mode=gym_render_mode)
    else:
        raise NotImplementedError(f"Environment '{args.env_name}' is not supported.")

    if args.video:
        save_dir = os.path.join(args.video_dir,
                                f"{args.env_name}_{args.algo}_seed{args.seed}")
        os.makedirs(save_dir, exist_ok=True)

        env = RecordVideo(env,
                          video_folder=save_dir,
                          episode_trigger=lambda ep: ep == 0,
                          name_prefix="eval_run")
        print(f"[INFO] MP4 will be written to {save_dir}")

    max_action = float(env.action_space.high[0])
    if isinstance(env, GymnasiumGoalEnv):
        controller_goal_dim = env.controller_goal_dim
        man_scale = env.manager_action_scale.copy()
        no_xy = True
    else:
        low = np.array((
            -10, -10, -0.5, -1, -1, -1, -1,
            -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3
        ))
        high = -low
        man_scale = (high - low) / 2
        if args.env_name == "AntFall":
            controller_goal_dim = 3
        else:
            controller_goal_dim = 2
        no_xy = True
        if args.absolute_goal:
            man_scale[0] = 30
            man_scale[1] = 30

    if args.absolute_goal:
        no_xy = False
    action_dim = env.action_space.shape[0]
    discrete = False

    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dim = state.shape[0]
    goal_dim = goal.shape[0] if goal is not None else 0

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

    checkpoint_spec = getattr(args, "checkpoint", None)
    checkpoint_suffix = None
    if isinstance(checkpoint_spec, str):
        cleaned = checkpoint_spec.strip()
        if cleaned and cleaned.lower() != "base":
            checkpoint_suffix = cleaned

    if args.load:
        try:
            manager_policy.load(args.model_dir, args.env_name, args.algo, suffix=checkpoint_suffix)
            controller_policy.load(args.model_dir, args.env_name, args.algo, suffix=checkpoint_suffix)
            tag = checkpoint_suffix or "base"
            print(f"Loaded checkpoint '{tag}'.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)
    
    if getattr(args, "sample_n", 0) and args.sample_n > 0:
        plots_dir = getattr(args, "plots_dir", "./plots")
        sample_and_plot_landings(env, args, manager_policy, controller_policy,
                                 controller_goal_dim, plots_dir)
        env.evaluate = False
        return None
    
        # NEW: heatmap mode (no st/gt required on CLI)
    if getattr(args, "heatmap", False):
        plots_dir = getattr(args, "plots_dir", "./plots")
        generate_variance_heatmap(env, args, manager_policy, controller_policy,
                                  controller_goal_dim, plots_dir)
        env.evaluate = False
        return None
    
    evaluate_policy(env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
                    1.0, args.manager_propose_freq, args.eval_episodes, args.render)
