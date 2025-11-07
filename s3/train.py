import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import glob
import os
import shutil
import numpy as np
import pandas as pd
from math import ceil

import s3.utils as utils
import s3.s3 as s3_algo
from s3.models import ANet
from envs import EnvWithGoal, GatherEnv
from envs.create_maze_env import create_maze_env
from envs.create_gather_env import create_gather_env
from s3.reach_net import ReachNet
import json

from s3.dispersion import dispersion_score
from envs.gymnasium_goal_env import GymnasiumGoalEnv, GYMNASIUM_ENV_SPECS


def archive_model_snapshots(model_dir: str, env_name: str, algo: str, tag: str) -> int:
    """Duplicate the latest manager/controller checkpoints with a suffix tag."""
    suffix = str(tag).strip()
    if not suffix:
        return 0

    pattern = os.path.join(model_dir, f"{env_name}_{algo}_*.pth")
    checkpoint_paths = glob.glob(pattern)
    if not checkpoint_paths:
        print(f"[WARN] No checkpoints found for {env_name}/{algo} while archiving '{suffix}'.")
        return 0

    for src in checkpoint_paths:
        base, ext = os.path.splitext(src)
        dst = f"{base}_{suffix}{ext}"
        shutil.copy2(src, dst)

    print(f"[INFO] Archived {len(checkpoint_paths)} checkpoints with tag '{suffix}' for {env_name}/{algo}.")
    return len(checkpoint_paths)

"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/train_hiro.py
"""


def evaluate_policy(env, env_name, manager_policy, controller_policy,
                    calculate_controller_reward, ctrl_rew_scale,
                    manager_propose_frequency=10, eval_idx=0, eval_episodes=5):
    print("Starting evaluation number {}...".format(eval_idx))
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
            reward = float(np.linalg.norm(subgoal - next_z, axis=-1) <= 1.414) * scale
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
            reward = float(np.linalg.norm(z + subgoal - next_z, axis=-1) <= 1.414) * scale
            return reward
    else:
        def controller_reward(z, subgoal, next_z, scale):
            z = z[:dims]
            next_z = next_z[:dims]
            reward = -np.linalg.norm(z + subgoal - next_z, axis=-1) * scale
            return reward

    return controller_reward


def update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
        optimizer_r, controller_goal_dim, device, args, episode_num):
    for traj in traj_buffer.get_trajectory():
        for i in range(len(traj)):
            for j in range(1, min(args.manager_propose_freq, len(traj) - i)):
                s1 = tuple(np.round(traj[i][:controller_goal_dim]).astype(np.int32))
                s2 = tuple(np.round(traj[i+j][:controller_goal_dim]).astype(np.int32))
                if s1 not in state_list:
                    state_list.append(s1)
                    state_dict[s1] = n_states
                    n_states += 1
                if s2 not in state_list:
                    state_list.append(s2)
                    state_dict[s2] = n_states
                    n_states += 1
                adj_mat[state_dict[s1], state_dict[s2]] = 1
                adj_mat[state_dict[s2], state_dict[s1]] = 1
    print("Explored states: {}".format(n_states))
    flags = np.ones((30, 30))
    for s in state_list:
        flags[int(s[0]), int(s[1])] = 0
    print(flags)
    if not args.load_adj_net:
        print("Training adjacency network...")
        utils.train_adj_net(a_net, state_list, adj_mat[:n_states, :n_states],
                            optimizer_r, args.r_margin_pos, args.r_margin_neg,
                            n_epochs=args.r_training_epochs, batch_size=args.r_batch_size,
                            device=device, verbose=False)

        if args.save_models:
            r_filename = os.path.join("./models", "{}_{}_a_network.pth".format(args.env_name, args.algo))
            torch.save(a_net.state_dict(), r_filename)
            print("----- Adjacency network {} saved. -----".format(episode_num))

    traj_buffer.reset()

    return n_states


def run_s3(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    model_dir = getattr(args, "model_dir", "./models")
    should_save_models = bool(
        getattr(args, "save_models", False) or
        getattr(args, "save_halfway_checkpoint", False) or
        getattr(args, "save_periodic", False)
    )

    if should_save_models and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.algo)):
        os.makedirs(os.path.join(args.log_dir, args.algo))
    output_dir = os.path.join(args.log_dir, args.algo)
    print("Logging in {}".format(output_dir))

    half_checkpoint_enabled = bool(getattr(args, "save_halfway_checkpoint", False))
    half_checkpoint_tag = getattr(args, "half_checkpoint_tag", "half")
    final_checkpoint_tag = getattr(args, "final_checkpoint_tag", "final")
    if half_checkpoint_enabled:
        half_checkpoint_step = max(1, int(args.max_timesteps / 2))
        half_checkpoint_saved = False
    else:
        half_checkpoint_step = None
        half_checkpoint_saved = True

    periodic_save_enabled = bool(getattr(args, "save_periodic", False))
    periodic_interval = 1_000_000
    next_periodic_checkpoint = periodic_interval if periodic_save_enabled else None
    periodic_checkpoint_index = 1

    if args.env_name == "AntGather":
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    elif args.env_name in ["AntMaze", "AntMazeSparse", "AntPush", "AntFall"]:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    elif args.env_name in GYMNASIUM_ENV_SPECS:
        env = GymnasiumGoalEnv(args.env_name, seed=args.seed)
    else:
        raise NotImplementedError(f"Environment '{args.env_name}' is not supported.")

    max_action = float(env.action_space.high[0])
    policy_noise = 0.2
    noise_clip = 0.5

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

    obs = env.reset()

    goal = obs["desired_goal"]
    state = obs["observation"]

    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.algo))
    torch.cuda.set_device(args.gid)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = "{}_{}_{}".format(args.env_name, args.algo, args.seed)
    output_data = {"frames": [], "reward": [], "dist": []}    
    if args.enable_transformer_logging:
        env_log = [] # Akshay's transformer piece

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    state_dim = state.shape[0]
    goal_dim = goal.shape[0] if goal is not None else 0

    controller_policy = s3_algo.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=args.ctrl_act_lr,
        critic_lr=args.ctrl_crit_lr,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=policy_noise,
        noise_clip=noise_clip
    )

    manager_policy = s3_algo.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=args.man_act_lr,
        critic_lr=args.man_crit_lr,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal
    )

    freeze_worker = bool(getattr(args, "freeze_worker", False))
    controller_updates_enabled = not freeze_worker
    if freeze_worker:
        worker_model_dir = getattr(args, "worker_model_dir", model_dir)
        worker_algo = getattr(args, "worker_algo", None) or args.algo
        worker_env_name = getattr(args, "worker_env_name", None) or args.env_name
        worker_suffix = getattr(args, "worker_checkpoint_tag", None)

        try:
            controller_policy.load(worker_model_dir, worker_env_name, worker_algo, suffix=worker_suffix)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to load frozen worker from '{worker_model_dir}' "
                f"for env '{worker_env_name}' and algo '{worker_algo}'."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error while loading frozen worker from '{worker_model_dir}'."
            ) from exc

        for module in (controller_policy.actor, controller_policy.actor_target,
                       controller_policy.critic, controller_policy.critic_target):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        suffix_msg = f" (checkpoint suffix '{worker_suffix}')" if worker_suffix else ""
        print(f"[INFO] Loaded frozen worker from {worker_model_dir} "
              f"using env '{worker_env_name}', algo '{worker_algo}'{suffix_msg}.")

        if (model_dir == worker_model_dir and
                worker_env_name == args.env_name and
                worker_algo == args.algo and
                worker_suffix is None):
            print("[WARN] Checkpoints will be saved to the same directory as the frozen worker. "
                  "Use --model_dir to avoid overwriting the pretrained weights.")

    calculate_controller_reward = get_reward_function(
        controller_goal_dim, absolute_goal=args.absolute_goal, binary_reward=args.binary_int_reward)
    if args.noise_type == "ou":
        man_noise = utils.OUNoise(state_dim, sigma=args.man_noise_sigma)
        ctrl_noise = utils.OUNoise(action_dim, sigma=args.ctrl_noise_sigma)

    elif args.noise_type == "normal":
        man_noise = utils.NormalNoise(sigma=args.man_noise_sigma)
        ctrl_noise = utils.NormalNoise(sigma=args.ctrl_noise_sigma)

    man_noise_sigma_init = float(args.man_noise_sigma)
    ctrl_noise_sigma_init = float(args.ctrl_noise_sigma)
    man_noise_sigma_final = args.man_noise_sigma_final if args.man_noise_sigma_final is not None else man_noise_sigma_init
    ctrl_noise_sigma_final = args.ctrl_noise_sigma_final if args.ctrl_noise_sigma_final is not None else ctrl_noise_sigma_init
    noise_anneal_start = max(0.0, float(getattr(args, "noise_anneal_start", 0.0)))
    noise_anneal_steps = max(0.0, float(getattr(args, "noise_anneal_steps", 0.0)))

    def _anneal_sigma(initial, final, step):
        if final is None or noise_anneal_steps <= 0 or initial == final:
            return initial
        if step <= noise_anneal_start:
            return initial
        progress = min(1.0, (step - noise_anneal_start) / noise_anneal_steps)
        if progress <= 0:
            return initial
        if initial > 0 and final > 0:
            ratio = max(final, 1e-8) / max(initial, 1e-8)
            sigma = initial * (ratio ** progress)
        else:
            sigma = initial + (final - initial) * progress
        if progress >= 1.0:
            return final
        return float(sigma)

    def _maybe_update_noise(step):
        if args.noise_type != "normal":
            return
        if man_noise is not None:
            man_noise.sigma = _anneal_sigma(man_noise_sigma_init, man_noise_sigma_final, step)
        if ctrl_noise is not None:
            ctrl_noise.sigma = _anneal_sigma(ctrl_noise_sigma_init, ctrl_noise_sigma_final, step)

    manager_buffer = utils.ReplayBuffer(maxsize=args.man_buffer_size)
    controller_buffer = utils.ReplayBuffer(maxsize=args.ctrl_buffer_size)
    reach_buffer = utils.ReachReplayBuffer(maxsize=args.reach_buffer_size)
    # Track reach-net readiness for shaping warm-up
    reach_training_rounds = 0
    reach_ready = False
    balance_warmup_start_step = None
    reach_warmup_samples = max(0, int(getattr(args, "reach_warmup_samples", 0)))
    reach_warmup_rounds = max(0, int(getattr(args, "reach_warmup_rounds", 0)))

    reach_nll_running = None
    reach_nll_update_count = 0

    fast_mode = bool(getattr(args, "fast_mode", False))
    if fast_mode:
        args.use_adj_net = False
        args.enable_transformer_logging = False
        args.log_dispersion_stats = False

    # Initialize reachability network
    reach_net = ReachNet(in_dim=controller_goal_dim*2,   # [s_xy , g_xy]
                     out_dim=controller_goal_dim, n_mix=args.n_mix).to(device)

    reach_optim = torch.optim.Adam(reach_net.parameters(), lr=1e-3)
    # Initialize adjacency matrix and adjacency network
    use_adj_net = bool(getattr(args, "use_adj_net", True))
    n_states = 0
    state_list = []
    state_dict = {}
    if use_adj_net:
        adj_mat = np.diag(np.ones(1500, dtype=np.uint8))
        traj_buffer = utils.TrajectoryBuffer(capacity=args.traj_buffer_size)
        a_net = ANet(controller_goal_dim, args.r_hidden_dim, args.r_embedding_dim)
        if args.load_adj_net:
            print("Loading adjacency network...")
            a_net.load_state_dict(torch.load("./models/a_network.pth"))
        a_net.to(device)
        optimizer_r = optim.Adam(a_net.parameters(), lr=args.lr_r)
    else:
        adj_mat = None
        traj_buffer = None
        a_net = None
        optimizer_r = None

    if args.load:
        try:
            manager_policy.load(model_dir, args.env_name, args.algo)
            if not freeze_worker:
                controller_policy.load(model_dir, args.env_name, args.algo)
            print("Loaded successfully.")
            just_loaded = True
        except Exception as e:
            just_loaded = False
            print(e, "Loading failed.")
    else:
        just_loaded = False

    # Logging Parameters
    total_timesteps = 0
    timesteps_since_eval = 0
    timesteps_since_manager = 0
    episode_timesteps = 0
    timesteps_since_subgoal = 0
    episode_num = 0
    done = True
    evaluations = []

    args.man_rew_scale = max(0.0, min(1.0, float(getattr(args, "man_rew_scale", 1.0))))
    if not hasattr(args, "man_env_scale"):
        args.man_env_scale = 0.1

    ext_reward = 0
    man_ctrl_rew_balance = 0.0
    # Train
    while total_timesteps < args.max_timesteps:
        _maybe_update_noise(total_timesteps)
        if done:
            if total_timesteps != 0 and not just_loaded:
                if episode_num % 10 == 0:
                    print("Episode {}".format(episode_num))
                # Train controller unless frozen
                if controller_updates_enabled:
                    ctrl_act_loss, ctrl_crit_loss = controller_policy.train(
                        controller_buffer,
                        episode_timesteps,
                        batch_size=args.ctrl_batch_size,
                        discount=args.ctrl_discount,
                        tau=args.ctrl_soft_sync_rate,
                    )
                    if episode_num % 10 == 0:
                        print("Controller actor loss: {:.3f}".format(ctrl_act_loss))
                        print("Controller critic loss: {:.3f}".format(ctrl_crit_loss))
                    writer.add_scalar("data/controller_actor_loss", ctrl_act_loss, total_timesteps)
                    writer.add_scalar("data/controller_critic_loss", ctrl_crit_loss, total_timesteps)
                elif episode_num % 10 == 0:
                    print("Controller frozen: skipping optimisation step.")


                writer.add_scalar("data/controller_ep_rew", episode_reward, total_timesteps)
                writer.add_scalar("data/manager_ep_rew", manager_transition[4], total_timesteps)
                writer.add_scalar("data/man_env_scale", args.man_env_scale, total_timesteps)
                writer.add_scalar("data/man_intrinsic_scale", args.man_rew_scale, total_timesteps)
                writer.add_scalar("data/ext_reward", ext_reward, total_timesteps)

                writer.add_scalar("data/var_bonus", var_bonus, total_timesteps)
                writer.add_scalar("data/man_ctrl_rew_balance", man_ctrl_rew_balance, total_timesteps)
                writer.add_scalar("data/reach_ready", float(reach_ready), total_timesteps)
                if args.log_dispersion_stats:
                    writer.add_scalar("tsd/trace",       disp_stats.get("trace", float("nan")),       total_timesteps)
                    writer.add_scalar("tsd/logdet",      disp_stats.get("logdet", float("nan")),      total_timesteps)
                    writer.add_scalar("tsd/maxeig",      disp_stats.get("maxeig", float("nan")),      total_timesteps)
                    writer.add_scalar("tsd/anisotropy",  disp_stats.get("anisotropy", float("nan")),  total_timesteps)
                    writer.add_scalar("tsd/dir_var",     disp_stats.get("dir_var", float("nan")),     total_timesteps)
                    writer.add_scalar("tsd/perp_var",    disp_stats.get("perp_var", float("nan")),    total_timesteps)
                    writer.add_scalar("tsd/w2",          disp_stats.get("w2", float("nan")),          total_timesteps)
                    writer.add_scalar("tsd/mi_proxy",    disp_stats.get("mi_proxy", float("nan")),    total_timesteps)
                    if "chance_eps" in disp_stats:
                        writer.add_scalar(f"tsd/chance_eps@{args.disp_eps}", disp_stats["chance_eps"], total_timesteps)
                    if "cvar" in disp_stats:
                        writer.add_scalar(f"tsd/cvar@{args.disp_alpha}", disp_stats["cvar"], total_timesteps)

                # landing error (empirical) for reliability tracking
                if not args.absolute_goal:
                    g_abs = state[:controller_goal_dim] + subgoal[:controller_goal_dim]
                else:
                    g_abs = subgoal[:controller_goal_dim]
                landing_err = float(np.linalg.norm(next_state[:controller_goal_dim] - g_abs))
                writer.add_scalar("tsd/landing_error", landing_err, total_timesteps)
                
                # Train manager
                if timesteps_since_manager >= args.train_manager_freq:
                    timesteps_since_manager = 0
                    r_margin = (args.r_margin_pos + args.r_margin_neg) / 2

                    train_kwargs = dict(
                        controller_policy=controller_policy,
                        replay_buffer=manager_buffer,
                        iterations=ceil(episode_timesteps/args.train_manager_freq),
                        batch_size=args.man_batch_size,
                        discount=args.man_discount,
                        tau=args.man_soft_sync_rate,
                        a_net=a_net,
                        r_margin=r_margin,
                        reach_net=reach_net,
                        phi_device=device,
                        args=args,
                        goal_dim=controller_goal_dim,
                        phi_samples=1,
                        intrinsic_scale=args.man_rew_scale,
                        intrinsic_mix=man_ctrl_rew_balance,
                    )
                    if use_adj_net:
                        man_act_loss, man_crit_loss, man_goal_loss = manager_policy.train(**train_kwargs)
                    else:
                        man_act_loss, man_crit_loss = manager_policy.train(**train_kwargs)
                        man_goal_loss = None

                    writer.add_scalar("data/manager_actor_loss", man_act_loss, total_timesteps)
                    writer.add_scalar("data/manager_critic_loss", man_crit_loss, total_timesteps)
                    if man_goal_loss is not None:
                        writer.add_scalar("data/manager_goal_loss", man_goal_loss, total_timesteps)

                    phi_stats = getattr(manager_policy, "latest_phi_stats", None)
                    if phi_stats is not None:
                        writer.add_scalar("tsd/phi_start", phi_stats["phi_start_mean"], total_timesteps)
                        writer.add_scalar("tsd/phi_end", phi_stats["phi_end_mean"], total_timesteps)
                        writer.add_scalar("tsd/pbrs_term", phi_stats["pbrs_mean"], total_timesteps)

                    if episode_num % 10 == 0:
                        print("Manager actor loss: {:.3f}".format(man_act_loss))
                        print("Manager critic loss: {:.3f}".format(man_crit_loss))
                        if man_goal_loss is not None:
                            print("Manager goal loss: {:.3f}".format(man_goal_loss))

                if total_timesteps % 10_000 == 0 and len(reach_buffer) > 5000:
                    reach_updates = 100
                    reach_batch = 512
                    total_samples = reach_updates * reach_batch

                    s0, g, sT = reach_buffer.sample(total_samples)
                    s0 = s0.astype(np.float32, copy=False)
                    g = g.astype(np.float32, copy=False)
                    sT = sT.astype(np.float32, copy=False)

                    s0_t = torch.from_numpy(s0).to(device)
                    g_t = torch.from_numpy(g).to(device)
                    sT_t = torch.from_numpy(sT).to(device)

                    reach_inputs = torch.cat((s0_t, g_t), dim=1).view(reach_updates, reach_batch, -1)
                    reach_targets = sT_t.view(reach_updates, reach_batch, -1)

                    reach_nll_batches = []
                    for inp_batch, tgt_batch in zip(reach_inputs, reach_targets):
                        # Reuse device tensors to avoid repeated host-to-device copies.
                        pi, mu, logS = reach_net(inp_batch)
                        var = torch.exp(logS).clamp_min(1e-3)
                        log_var = torch.log(var)
                        exp_term = -0.5 * ((tgt_batch[:, None, :] - mu) ** 2 / var).sum(-1)
                        log_prob = torch.log(pi + 1e-9) + exp_term \
                                - 0.5 * log_var.sum(-1) - 0.5 * mu.size(-1) * np.log(2 * np.pi)
                        nll = -torch.logsumexp(log_prob, -1).mean()

                        reach_optim.zero_grad()
                        nll.backward()
                        reach_optim.step()
                        reach_nll_batches.append(nll.detach().cpu())

                    if reach_nll_batches:
                        reach_nll_mean = torch.stack(reach_nll_batches).mean()
                        reach_nll_update_count += 1
                        if reach_nll_running is None:
                            reach_nll_running = reach_nll_mean.item()
                        else:
                            reach_nll_running += (
                                reach_nll_mean.item() - reach_nll_running
                            ) / reach_nll_update_count

                        writer.add_scalar("data/reach_nll", reach_nll_mean.item(), total_timesteps)
                        writer.add_scalar("data/reach_nll_running", reach_nll_running, total_timesteps)

                    reach_training_rounds += 1
                    if (not reach_ready and
                            len(reach_buffer) >= reach_warmup_samples and
                            reach_training_rounds >= reach_warmup_rounds):
                        reach_ready = True
                        balance_warmup_start_step = total_timesteps

                    
                # Evaluate
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval = 0
                    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish =\
                        evaluate_policy(env, args.env_name, manager_policy, controller_policy,
                            calculate_controller_reward, args.ctrl_rew_scale, args.manager_propose_freq,
                            len(evaluations))

                    writer.add_scalar("eval/avg_ep_rew", avg_ep_rew, total_timesteps)
                    writer.add_scalar("eval/avg_controller_rew", avg_controller_rew, total_timesteps)

                    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
                    output_data["frames"].append(total_timesteps)
                    if args.env_name == "AntGather":
                        output_data["reward"].append(avg_ep_rew)
                    else:
                        output_data["reward"].append(avg_env_finish)
                        writer.add_scalar("eval/avg_steps_to_finish", avg_steps, total_timesteps)
                        writer.add_scalar("eval/perc_env_goal_achieved", avg_env_finish, total_timesteps)
                    output_data["dist"].append(-avg_controller_rew)

                    if args.save_models:
                        controller_policy.save(model_dir, args.env_name, args.algo)
                        manager_policy.save(model_dir, args.env_name, args.algo)

                if use_adj_net and traj_buffer.full():
                     n_states = update_amat_and_train_anet(n_states, adj_mat, state_list, state_dict, a_net, traj_buffer,
                        optimizer_r, controller_goal_dim, device, args, episode_num)

                if len(manager_transition[-2]) != 1:                    
                    manager_transition[1] = state
                    manager_transition[5] = float(True)

                    manager_buffer.add(manager_transition)

            obs = env.reset()
            goal = obs["desired_goal"]
            state = obs["observation"]
            if use_adj_net:
                traj_buffer.create_new_trajectory()
                traj_buffer.append(state)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            just_loaded = False
            episode_num += 1

            subgoal = manager_policy.sample_goal(state, goal)
            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
            else:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

            timesteps_since_subgoal = 0
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

        action = controller_policy.select_action(state, subgoal)
        action = ctrl_noise.perturb_action(action, -max_action, max_action)
        action_copy = action.copy()

        next_tup, manager_reward, done, _ = env.step(action_copy)
        next_goal = next_tup["desired_goal"]
        next_state = next_tup["observation"]
        if args.enable_transformer_logging:
            env_log.append({
                "observation": state.tolist(),
                "action": action_copy.tolist(),
                "reward": manager_reward,  # or `manager_reward` or `controller_reward` based on your use
                "done": done,
                "desired_goal": goal.tolist(),
                "subgoal": subgoal.tolist(),
                "next_observation": next_state.tolist(),
                "next_goal": next_goal.tolist(),

            }) # Akshay's transformer piece

        # Use man_ctrl_rew_balance to balance manager and controller rewards

        if reach_ready:
            if balance_warmup_start_step is None:
                balance_warmup_start_step = total_timesteps
            elapsed_since_warmup = max(0, total_timesteps - balance_warmup_start_step)
            man_ctrl_rew_balance = anneal_balance(
                elapsed_since_warmup,
                args.man_ctrl_rew_balance_start,
                args.man_ctrl_rew_balance_end,
                args.man_ctrl_rew_balance_steps)
        else:
            man_ctrl_rew_balance = 0.0
        
        with torch.no_grad():
            inp = torch.from_numpy(
                    np.hstack([state[:controller_goal_dim], subgoal])
                ).float().unsqueeze(0).to(device)
            pi, mu, logS = reach_net(inp)
            
            # pi  = pi.unsqueeze(-1)                 # ► (1 , M , 1)


            disp_score, disp_stats = dispersion_score(
                pi, mu, logS,
                state_np=state, subgoal_np=subgoal,
                d=controller_goal_dim, args=args, device=device,
                collect_stats=args.log_dispersion_stats,
            )
            
            # # law of total variance  Var[X] = E[Σ+μ²] − (E[μ])²
            # mu_mix = (pi * mu).sum(1)              # (1 , D)
            # var_mix = (pi * (torch.exp(logS) + mu.pow(2))).sum(1) - mu_mix.pow(2)

            # reach_var = var_mix.sum().item()
            
        var_bonus = -args.man_rew_scale * float(disp_score)

        controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        env_reward = manager_reward
        intrinsic_reward = var_bonus
        if man_ctrl_rew_balance > 0.0:
            manager_reward = (
                (1.0 - man_ctrl_rew_balance) * env_reward +
                man_ctrl_rew_balance * intrinsic_reward
            )
        else:
            manager_reward = env_reward

        ext_reward += env_reward * args.man_env_scale
        manager_transition[4] += manager_reward * args.man_env_scale
        manager_transition[-1].append(action)

        
        manager_transition[-2].append(next_state)
        if use_adj_net:
            traj_buffer.append(next_state)

        reach_buffer.add((state[:controller_goal_dim],
                                  subgoal.copy(),
                                  next_state[:controller_goal_dim]))
        
        # controller_reward = calculate_controller_reward(state, subgoal, next_state, args.ctrl_rew_scale)
        subgoal = controller_policy.subgoal_transition(state, subgoal, next_state)
        
        
        
        controller_goal = subgoal
        episode_reward += controller_reward

        if args.inner_dones:
            ctrl_done = done or timesteps_since_subgoal % args.manager_propose_freq == 0
        else:
            ctrl_done = done

        controller_buffer.add(
            (state, next_state, controller_goal, action, controller_reward, float(ctrl_done), [], []))

        state = next_state
        goal = next_goal

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        timesteps_since_manager += 1
        timesteps_since_subgoal += 1

        if half_checkpoint_enabled and not half_checkpoint_saved and \
                total_timesteps >= half_checkpoint_step:
            controller_policy.save(model_dir, args.env_name, args.algo)
            manager_policy.save(model_dir, args.env_name, args.algo)
            archive_model_snapshots(model_dir, args.env_name, args.algo, half_checkpoint_tag)
            half_checkpoint_saved = True
            print(f"[INFO] Halfway checkpoint saved at {total_timesteps} steps.")

        if periodic_save_enabled and next_periodic_checkpoint is not None:
            while total_timesteps >= next_periodic_checkpoint:
                controller_policy.save(model_dir, args.env_name, args.algo)
                manager_policy.save(model_dir, args.env_name, args.algo)
                periodic_tag = f"{periodic_checkpoint_index}M"
                archive_model_snapshots(model_dir, args.env_name, args.algo, periodic_tag)
                print(f"[INFO] Periodic checkpoint '{periodic_tag}' saved at {total_timesteps} steps.")
                periodic_checkpoint_index += 1
                next_periodic_checkpoint += periodic_interval

        if timesteps_since_subgoal % args.manager_propose_freq == 0:
            manager_transition[1] = state
            manager_transition[5] = float(done)
            manager_buffer.add(manager_transition)
            subgoal = manager_policy.sample_goal(state, goal)

            if not args.absolute_goal:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=-man_scale[:controller_goal_dim], max_action=man_scale[:controller_goal_dim])
            else:
                subgoal = man_noise.perturb_action(subgoal,
                    min_action=np.zeros(controller_goal_dim), max_action=2*man_scale[:controller_goal_dim])

            timesteps_since_subgoal = 0
            manager_transition = [state, None, goal, subgoal, 0, False, [state], []]

            reach_buffer.add((
            state[:controller_goal_dim].copy(),     # s_t  (XY)
            manager_transition[3].copy(),           # g_t  (the subgoal we just executed)
            state[:controller_goal_dim].copy()      # s_{t+k}  (current state)
            ))



    # Final evaluation
    avg_ep_rew, avg_controller_rew, avg_steps, avg_env_finish = evaluate_policy(
        env, args.env_name, manager_policy, controller_policy, calculate_controller_reward,
        args.ctrl_rew_scale, args.manager_propose_freq, len(evaluations))
    evaluations.append([avg_ep_rew, avg_controller_rew, avg_steps])
    output_data["frames"].append(total_timesteps)
    if args.env_name == 'AntGather':
        output_data["reward"].append(avg_ep_rew)
    else:
        output_data["reward"].append(avg_env_finish)
    output_data["dist"].append(-avg_controller_rew)

    if args.save_models or half_checkpoint_enabled:
        controller_policy.save(model_dir, args.env_name, args.algo)
        manager_policy.save(model_dir, args.env_name, args.algo)

    if half_checkpoint_enabled:
        archive_model_snapshots(model_dir, args.env_name, args.algo, final_checkpoint_tag)

    writer.close()

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(os.path.join("./results", file_name+".csv"), float_format="%.4f", index=False)
    if args.enable_transformer_logging:
        with open(os.path.join("./results", file_name + "_envlog.json"), "w") as f: # Akshay's transformer piece
            json.dump(env_log, f)
        print("Results saved to ./results/{}.csv".format(file_name))
        print("Training finished.")

def anneal_balance(step: int, start: float, min_b: float, total_steps: int) -> float:
    if total_steps <= 0:
        return start                # disabled
    frac = min(1.0, step / total_steps)   # 0 → 1
    return start - (start - min_b) * frac
