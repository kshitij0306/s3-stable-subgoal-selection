'''Visualization utilities for S3 subgoal diagnostics.'''

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import art3d

import numpy as np
import torch

import s3.s3 as s3_algo
from s3.dispersion import dispersion_score
from s3.reach_net import ReachNet

from envs import EnvWithGoal, GatherEnv
from envs.create_gather_env import create_gather_env
from envs.create_maze_env import create_maze_env
from envs import maze_env_utils


@dataclass
class MazeMeta:
    structure: Sequence[Sequence[object]]
    size_scaling: float
    torso_offset: Tuple[float, float]
    height: float
    elevated: bool


@dataclass
class Segment:
    index: int
    rel_goal: np.ndarray
    abs_goal: np.ndarray
    dispersion: Optional[float]
    positions: np.ndarray


@dataclass
class EpisodeTrace:
    tag: Optional[str]
    episode_index: int
    start_xy: Optional[np.ndarray]
    desired_goal: Optional[np.ndarray]
    positions: np.ndarray
    segments: List[Segment]


def _parse_checkpoint_list(spec: Optional[str]) -> List[Optional[str]]:
    if not spec:
        return [None]
    tags: List[Optional[str]] = []
    for item in spec.split(','):
        tag = item.strip()
        if not tag or tag.lower() in {'base', 'none', 'latest'}:
            tags.append(None)
        else:
            tags.append(tag)
    return tags


def _compute_manager_scale(absolute_goal: bool) -> Tuple[np.ndarray, bool]:
    low = np.array(
        (-10, -10, -0.5, -1, -1, -1, -1,
         -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3),
        dtype=np.float32,
    )
    high = -low
    man_scale = (high - low) / 2.0
    if absolute_goal:
        man_scale[0] = 30.0
        man_scale[1] = 30.0
        no_xy = False
    else:
        no_xy = True
    return man_scale, no_xy


def _create_env(args):
    if args.env_name == 'AntGather':
        env = GatherEnv(create_gather_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    elif args.env_name in {'AntMaze', 'AntMazeSparse', 'AntPush', 'AntFall'}:
        env = EnvWithGoal(create_maze_env(args.env_name, args.seed), args.env_name)
        env.seed(args.seed)
    else:
        raise NotImplementedError(f'Unsupported env: {args.env_name}')

    obs = env.reset()
    state = obs['observation']
    goal = obs.get('desired_goal')

    controller_goal_dim = 3 if args.env_name == 'AntFall' else 2
    state_dim = state.shape[0]
    goal_dim = goal.shape[0] if goal is not None else 0
    max_action = float(env.action_space.high[0])
    action_dim = env.action_space.shape[0]
    man_scale, no_xy = _compute_manager_scale(args.absolute_goal)

    maze_meta: Optional[MazeMeta] = None
    base_env = getattr(env, 'base_env', None)
    if base_env is not None and hasattr(base_env, 'MAZE_STRUCTURE'):
        maze_meta = MazeMeta(
            structure=base_env.MAZE_STRUCTURE,
            size_scaling=float(base_env.MAZE_SIZE_SCALING),
            torso_offset=(float(base_env._init_torso_x), float(base_env._init_torso_y)),
            height=float(base_env.MAZE_HEIGHT),
            elevated=bool(getattr(base_env, 'elevated', False)),
        )

    return env, state_dim, goal_dim, controller_goal_dim, max_action, action_dim, man_scale, no_xy, maze_meta


def _init_policies(state_dim, goal_dim, controller_goal_dim, action_dim, max_action, man_scale, no_xy, args):
    torch.cuda.set_device(args.gid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    controller = s3_algo.Controller(
        state_dim=state_dim,
        goal_dim=controller_goal_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor_lr=0,
        critic_lr=0,
        no_xy=no_xy,
        absolute_goal=args.absolute_goal,
        policy_noise=0,
        noise_clip=0,
    )

    manager = s3_algo.Manager(
        state_dim=state_dim,
        goal_dim=goal_dim,
        action_dim=controller_goal_dim,
        actor_lr=0,
        critic_lr=0,
        candidate_goals=args.candidate_goals,
        correction=not args.no_correction,
        scale=man_scale,
        goal_loss_coeff=args.goal_loss_coeff,
        absolute_goal=args.absolute_goal,
    )
    for module in (manager.actor, manager.actor_target, manager.critic, manager.critic_target):
        module.eval()

    for module in (controller.actor, controller.actor_target, controller.critic, controller.critic_target):
        module.eval()

    return manager, controller, device


def _load_checkpoint(manager_policy, controller_policy, model_dir: Path, env_name: str,
                     algo: str, suffix: Optional[str], device: torch.device) -> None:
    suffix_part = f'_{suffix}' if suffix else ''
    specs = [
        (manager_policy.actor, f'{env_name}_{algo}_ManagerActor{suffix_part}.pth'),
        (manager_policy.critic, f'{env_name}_{algo}_ManagerCritic{suffix_part}.pth'),
        (manager_policy.actor_target, f'{env_name}_{algo}_ManagerActorTarget{suffix_part}.pth'),
        (manager_policy.critic_target, f'{env_name}_{algo}_ManagerCriticTarget{suffix_part}.pth'),
        (controller_policy.actor, f'{env_name}_{algo}_ControllerActor{suffix_part}.pth'),
        (controller_policy.critic, f'{env_name}_{algo}_ControllerCritic{suffix_part}.pth'),
        (controller_policy.actor_target, f'{env_name}_{algo}_ControllerActorTarget{suffix_part}.pth'),
        (controller_policy.critic_target, f'{env_name}_{algo}_ControllerCriticTarget{suffix_part}.pth'),
    ]

    for module, filename in specs:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing checkpoint file: {path}')
        state = torch.load(path, map_location=device)
        module.load_state_dict(state)


def _resolve_reach_path(args, suffix: Optional[str]) -> Optional[Path]:
    suffix_part = f'_{suffix}' if suffix else ''
    candidate_specs: List[str] = []
    template = getattr(args, 'reach_path', None)
    if template:
        formatted = template.format(
            env=args.env_name,
            algo=args.algo,
            suffix=suffix_part,
            tag=suffix or '',
        )
        candidate_specs.append(formatted)

    candidate_specs.append(f'{args.env_name}_{args.algo}_ReachNet{suffix_part}.pth')
    candidate_specs.append(f'{args.env_name}_{args.algo}_reach_net{suffix_part}.pth')

    for spec in candidate_specs:
        path = Path(spec)
        if path.is_file():
            return path
        if not path.is_absolute():
            alt = Path(args.model_dir) / path
            if alt.is_file():
                return alt
    return None


def _maybe_load_reach_net(args, controller_goal_dim: int, device: torch.device,
                          suffix: Optional[str]) -> Optional[ReachNet]:
    if getattr(args, 'disable_reach', False):
        return None

    path = _resolve_reach_path(args, suffix)
    if path is None:
        return None

    reach_net = ReachNet(
        in_dim=controller_goal_dim * 2,
        out_dim=controller_goal_dim,
        n_mix=args.n_mix,
    )
    state = torch.load(path, map_location=device)
    reach_net.load_state_dict(state)
    reach_net.to(device)
    reach_net.eval()
    return reach_net


def _goal_array(goal, goal_dim: int) -> np.ndarray:
    if goal_dim == 0:
        return np.zeros((0,), dtype=np.float32)
    if goal is None:
        return np.zeros((goal_dim,), dtype=np.float32)
    return np.asarray(goal, dtype=np.float32)


def _compute_dispersion(reach_net: Optional[ReachNet], state: np.ndarray, rel_goal: np.ndarray,
                        controller_goal_dim: int, args, device: torch.device) -> Optional[float]:
    if reach_net is None:
        return None

    prefix = state[:controller_goal_dim]
    goal_part = rel_goal[:controller_goal_dim]
    inp = np.hstack([prefix, goal_part]).astype(np.float32, copy=False)

    with torch.no_grad():
        tensor = torch.from_numpy(inp).unsqueeze(0).to(device)
        pi, mu, logS = reach_net(tensor)
        score, _ = dispersion_score(
            pi,
            mu,
            logS,
            state_np=state,
            subgoal_np=rel_goal,
            d=controller_goal_dim,
            args=args,
            device=device,
            collect_stats=False,
        )
    return float(score)


def _start_segment(index: int, state: np.ndarray, rel_goal: np.ndarray, controller_goal_dim: int,
                   args, reach_net: Optional[ReachNet], device: torch.device):
    rel_goal_vec = np.asarray(rel_goal, dtype=np.float32)
    state_prefix = state[:controller_goal_dim]
    if args.absolute_goal:
        abs_goal = rel_goal_vec[:controller_goal_dim].copy()
    else:
        abs_goal = state_prefix + rel_goal_vec[:controller_goal_dim]
    dispersion = _compute_dispersion(reach_net, state, rel_goal_vec, controller_goal_dim, args, device)
    return {
        'index': index,
        'rel_goal': rel_goal_vec,
        'abs_goal': abs_goal,
        'dispersion': dispersion,
        'positions': [state_prefix.copy()],
    }


def _finalise_segment(data: dict) -> Segment:
    positions = np.asarray(data['positions'], dtype=np.float32)
    return Segment(
        index=int(data['index']),
        rel_goal=np.asarray(data['rel_goal'], dtype=np.float32),
        abs_goal=np.asarray(data['abs_goal'], dtype=np.float32),
        dispersion=data['dispersion'],
        positions=positions,
    )


def _collect_episode(env, manager_policy, controller_policy, controller_goal_dim: int, goal_dim: int,
                     args, reach_net: Optional[ReachNet], device: torch.device, tag: Optional[str],
                     episode_index: int, start_xy=None, desired_goal=None) -> EpisodeTrace:
    env.evaluate = True
    if start_xy is not None:
        try:
            obs = env.reset(np.asarray(start_xy, dtype=np.float32))
        except Exception:
            obs = env.reset()
    else:
        obs = env.reset()

    if desired_goal is not None and goal_dim:
        goal_override = np.asarray(desired_goal, dtype=np.float32)
        env.goal = goal_override.copy()
        env.desired_goal = goal_override.copy()
        obs['desired_goal'] = goal_override.copy()

    state = np.asarray(obs['observation'], dtype=np.float32)
    achieved = obs.get('achieved_goal')
    start_snapshot = np.asarray(achieved, dtype=np.float32) if achieved is not None else None
    goal_vec = _goal_array(obs.get('desired_goal'), goal_dim)

    episode_positions: List[np.ndarray] = [state[:controller_goal_dim].copy()]
    segments: List[Segment] = []

    rel_goal = manager_policy.sample_goal(state, goal_vec)
    segment_data = _start_segment(0, state, rel_goal, controller_goal_dim, args, reach_net, device)
    steps_since_subgoal = 0
    max_steps = int(args.max_steps)

    for _ in range(max_steps):
        action = controller_policy.select_action(state, rel_goal, evaluation=True)
        next_obs, _, done, _ = env.step(action)
        next_state = np.asarray(next_obs['observation'], dtype=np.float32)
        point = next_state[:controller_goal_dim].copy()
        segment_data['positions'].append(point)
        episode_positions.append(point)

        rel_goal = controller_policy.subgoal_transition(state, rel_goal, next_state)
        state = next_state
        goal_vec = _goal_array(next_obs.get('desired_goal'), goal_dim)
        steps_since_subgoal += 1

        boundary = done or steps_since_subgoal >= args.manager_propose_freq
        if boundary:
            segments.append(_finalise_segment(segment_data))
            if done:
                break
            segment_index = len(segments)
            rel_goal = manager_policy.sample_goal(state, goal_vec)
            segment_data = _start_segment(segment_index, state, rel_goal, controller_goal_dim, args, reach_net, device)
            steps_since_subgoal = 0

    last_index = segments[-1].index if segments else -1
    if segment_data['index'] != last_index and len(segment_data['positions']) > 1:
        segments.append(_finalise_segment(segment_data))

    positions = np.asarray(episode_positions, dtype=np.float32)
    desired_goal_vec = _goal_array(env.desired_goal if goal_dim else None, goal_dim) if goal_dim else None

    return EpisodeTrace(
        tag=tag if tag is None else str(tag),
        episode_index=episode_index,
        start_xy=start_snapshot.copy() if start_snapshot is not None else None,
        desired_goal=desired_goal_vec,
        positions=positions,
        segments=segments,
    )


def _stack_xy(arrays: Iterable[np.ndarray]) -> np.ndarray:
    mats: List[np.ndarray] = []
    for arr in arrays:
        if arr is None or not hasattr(arr, 'shape') or arr.size == 0:
            continue
        if arr.shape[1] < 2:
            continue
        mats.append(arr[:, :2])
    if not mats:
        return np.zeros((0, 2), dtype=np.float32)
    return np.vstack(mats)


def _stack_xyz(arrays: Iterable[np.ndarray]) -> np.ndarray:
    mats: List[np.ndarray] = []
    for arr in arrays:
        if arr is None or not hasattr(arr, 'shape') or arr.size == 0:
            continue
        if arr.shape[1] < 3:
            continue
        mats.append(arr[:, :3])
    if not mats:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(mats)


def _axis_limits(maze_meta: Optional[MazeMeta], xy: np.ndarray) -> Tuple[float, float, float, float]:
    if maze_meta is not None:
        width = len(maze_meta.structure[0]) * maze_meta.size_scaling
        height = len(maze_meta.structure) * maze_meta.size_scaling
        min_x = -maze_meta.torso_offset[0] - maze_meta.size_scaling / 2.0
        min_y = -maze_meta.torso_offset[1] - maze_meta.size_scaling / 2.0
        max_x = min_x + width
        max_y = min_y + height
        return min_x, max_x, min_y, max_y

    if xy.size:
        min_xy = xy.min(axis=0)
        max_xy = xy.max(axis=0)
        margin = 0.5
        return min_xy[0] - margin, max_xy[0] + margin, min_xy[1] - margin, max_xy[1] + margin

    return -5.0, 5.0, -5.0, 5.0


def _draw_maze_background_2d(ax, maze_meta: Optional[MazeMeta], alpha: float) -> None:
    if maze_meta is None:
        return

    size = maze_meta.size_scaling
    offset_x, offset_y = maze_meta.torso_offset

    for i, row in enumerate(maze_meta.structure):
        for j, cell in enumerate(row):
            cx = j * size - offset_x
            cy = i * size - offset_y
            lower_left = (cx - size / 2.0, cy - size / 2.0)
            if cell == 1:
                color = (0.25, 0.25, 0.25, alpha)
            elif cell == -1:
                color = (0.85, 0.85, 0.85, alpha)
            elif maze_env_utils.can_move(cell):
                color = (0.7, 0.2, 0.2, alpha)
            else:
                color = (0.92, 0.92, 0.92, alpha * 0.45)
            ax.add_patch(Rectangle(lower_left, size, size, facecolor=color, edgecolor='none'))

    ax.set_aspect('equal', adjustable='box')


def _add_box3d(ax, center: Tuple[float, float, float], half_extents: Tuple[float, float, float],
               color: Tuple[float, float, float], alpha: float = 1.0, edge_color: str = 'k',
               edge_width: float = 0.2) -> None:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    xs = [cx - hx, cx + hx]
    ys = [cy - hy, cy + hy]
    zs = [cz - hz, cz + hz]
    vertices = np.array(list(product(xs, ys, zs)))
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    poly3d = [[vertices[idx] for idx in face] for face in faces]
    collection = art3d.Poly3DCollection(poly3d, facecolors=[(*color, alpha)],
                                        edgecolor=edge_color, linewidth=edge_width)
    ax.add_collection3d(collection)


def _draw_maze_background_3d(ax, maze_meta: Optional[MazeMeta], alpha: float) -> None:
    if maze_meta is None:
        return

    size = maze_meta.size_scaling
    offset_x, offset_y = maze_meta.torso_offset
    platform_half = (size * 0.5, size * 0.5, maze_meta.height * size * 0.5)
    platform_center_z = maze_meta.height * size * 0.5
    block_center_z = maze_meta.height * size * 1.5

    for i, row in enumerate(maze_meta.structure):
        for j, cell in enumerate(row):
            cx = j * size - offset_x
            cy = i * size - offset_y

            if cell != -1 and maze_meta.elevated:
                base_color = (0.92, 0.92, 0.92)
                _add_box3d(ax, (cx, cy, platform_center_z), platform_half, base_color, alpha)

            if cell == 1:
                color = (0.25, 0.25, 0.25)
                _add_box3d(ax, (cx, cy, block_center_z), platform_half, color, 0.95)
            elif maze_env_utils.can_move(cell):
                color = (0.7, 0.2, 0.2)
                _add_box3d(ax, (cx, cy, block_center_z), platform_half, color, 0.9)


def _axis_limits_3d(maze_meta: Optional[MazeMeta], xyz: np.ndarray) -> Tuple[Tuple[float, float],
                                                                             Tuple[float, float],
                                                                             Tuple[float, float]]:
    margin = 0.5
    if maze_meta is not None:
        width = len(maze_meta.structure[0]) * maze_meta.size_scaling
        height = len(maze_meta.structure) * maze_meta.size_scaling
        min_x = -maze_meta.torso_offset[0] - maze_meta.size_scaling / 2.0
        min_y = -maze_meta.torso_offset[1] - maze_meta.size_scaling / 2.0
        max_x = min_x + width
        max_y = min_y + height
        min_z = 0.0
        max_z = maze_meta.height * maze_meta.size_scaling * (2.15 if maze_meta.elevated else 1.1)
        return ((min_x - margin, max_x + margin),
                (min_y - margin, max_y + margin),
                (min_z, max_z + margin))

    if xyz.size:
        min_xyz = xyz.min(axis=0)
        max_xyz = xyz.max(axis=0)
        return ((min_xyz[0] - margin, max_xyz[0] + margin),
                (min_xyz[1] - margin, max_xyz[1] + margin),
                (max(0.0, min_xyz[2] - margin), max_xyz[2] + margin))

    return ((-5.0, 5.0), (-5.0, 5.0), (0.0, 5.0))


def _landing_point(segment: Segment) -> Optional[np.ndarray]:
    if segment.positions.size == 0:
        return None
    return segment.positions[-1]


def _with_alpha(color: Tuple[float, float, float, float], alpha: float) -> Tuple[float, float, float, float]:
    if len(color) == 4:
        return (color[0], color[1], color[2], alpha)
    return to_rgba(color, alpha=alpha)


def _draw_landing_radius(ax, goal: np.ndarray, landing: np.ndarray,
                         color: Tuple[float, float, float, float], is_3d: bool, args) -> None:
    radius = np.linalg.norm(landing[:2] - goal[:2])
    if radius <= 1e-6:
        return
    alpha = getattr(args, 'subgoal_radius_alpha', 0.2)
    face_color = _with_alpha(color, alpha)
    circle = Circle((goal[0], goal[1]), radius, facecolor=face_color, edgecolor='none')
    if is_3d:
        ax.add_patch(circle)
        z_level = goal[2] if goal.shape[0] >= 3 else (landing[2] if landing.shape[0] >= 3 else 0.0)
        art3d.pathpatch_2d_to_3d(circle, z=z_level, zdir='z')
    else:
        ax.add_patch(circle)


def _plot_checkpoint_overlay(episodes: List[EpisodeTrace], maze_meta: Optional[MazeMeta],
                              args, out_path: Path, tag_label: str) -> None:
    is_3d = any(episode.positions.shape[1] >= 3 for episode in episodes)
    if is_3d:
        fig = plt.figure(figsize=(args.figure_size, args.figure_size))
        ax = fig.add_subplot(111, projection='3d')
        _draw_maze_background_3d(ax, maze_meta, args.floor_alpha)
    else:
        fig, ax = plt.subplots(figsize=(args.figure_size, args.figure_size))
        _draw_maze_background_2d(ax, maze_meta, args.floor_alpha)

    all_segments = [segment for episode in episodes for segment in episode.segments]
    total_segments = len(all_segments)
    cmap = cm.get_cmap(args.overlay_cmap, max(total_segments, 1))

    lines_2d = []
    line_colors: List[Tuple[float, float, float, float]] = []
    subgoals: List[np.ndarray] = []
    subgoal_colors: List[Tuple[float, float, float, float]] = []
    landings: List[np.ndarray] = []
    landing_colors: List[Tuple[float, float, float, float]] = []

    for idx, segment in enumerate(all_segments):
        coords = segment.positions[:, :3] if is_3d else segment.positions[:, :2]
        color = cmap(idx / max(total_segments - 1, 1))
        if coords.shape[0] >= 2:
            if is_3d:
                ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color,
                        linewidth=args.trace_width, alpha=args.trace_alpha)
            else:
                lines_2d.append(coords)
                line_colors.append(color)

        subgoals.append(segment.abs_goal)
        subgoal_colors.append(color)

        landing = _landing_point(segment)
        if landing is not None:
            landings.append(landing)
            landing_colors.append(color)
            _draw_landing_radius(ax, segment.abs_goal, landing, color, is_3d, args)

    if not is_3d and lines_2d:
        lc = LineCollection(lines_2d, colors=line_colors, linewidths=args.trace_width, alpha=args.trace_alpha)
        ax.add_collection(lc)

    if subgoals:
        if is_3d:
            xs = [float(goal[0]) for goal in subgoals]
            ys = [float(goal[1]) for goal in subgoals]
            zs = [float(goal[2]) if goal.shape[0] >= 3 else 0.0 for goal in subgoals]
            ax.scatter(xs, ys, zs, c=subgoal_colors, s=args.subgoal_marker_size,
                       marker='o', edgecolor='k', linewidth=0.4, depthshade=False,
                       label='Manager subgoal')
        else:
            xs = [float(goal[0]) for goal in subgoals]
            ys = [float(goal[1]) for goal in subgoals]
            ax.scatter(xs, ys, c=subgoal_colors, s=args.subgoal_marker_size,
                       marker='o', edgecolor='k', linewidth=0.4, label='Manager subgoal')

    if landings:
        landing_alpha = getattr(args, 'landing_alpha', 0.45)
        landing_rgba = [_with_alpha(color, landing_alpha) for color in landing_colors]
        if is_3d:
            xs = [float(point[0]) for point in landings]
            ys = [float(point[1]) for point in landings]
            zs = [float(point[2]) if point.shape[0] >= 3 else 0.0 for point in landings]
            ax.scatter(xs, ys, zs, c=landing_rgba, s=getattr(args, 'landing_marker_size', 32.0),
                       marker='o', edgecolor='none', depthshade=False, label='Landing state')
        else:
            xs = [float(point[0]) for point in landings]
            ys = [float(point[1]) for point in landings]
            ax.scatter(xs, ys, c=landing_rgba, s=getattr(args, 'landing_marker_size', 32.0),
                       marker='o', edgecolors='none', label='Landing state')

    if is_3d:
        paths_xyz = _stack_xyz([episode.positions for episode in episodes])
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = _axis_limits_3d(maze_meta, paths_xyz)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
    else:
        paths_xy = _stack_xy([episode.positions for episode in episodes])
        min_x, max_x, min_y, max_y = _axis_limits(maze_meta, paths_xy)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    if is_3d:
        starts = _stack_xyz([episode.positions[:1] for episode in episodes])
        if starts.size:
            ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2], c='k', marker='s', s=50,
                       depthshade=False, label='Episode start')
    else:
        starts = _stack_xy([episode.positions[:1] for episode in episodes])
        if starts.size:
            ax.scatter(starts[:, 0], starts[:, 1], c='k', marker='s', s=50, label='Episode start')

    desired = [episode.desired_goal for episode in episodes
               if episode.desired_goal is not None and episode.desired_goal.size >= (3 if is_3d else 2)]
    if desired:
        if is_3d:
            arr = np.stack([goal[:3] for goal in desired])
            ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c='gold', marker='*', s=110,
                       edgecolor='k', depthshade=False, label='Desired goal')
        else:
            arr = np.stack([goal[:2] for goal in desired])
            ax.scatter(arr[:, 0], arr[:, 1], c='gold', marker='*', s=110,
                       edgecolor='k', label='Desired goal')

    ax.set_title(f'{args.env_name} – {args.algo} ({tag_label})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if is_3d:
        ax.set_zlabel('z')
        elev = getattr(args, 'view_elev', 40.0)
        azim = getattr(args, 'view_azim', -90.0)
        ax.view_init(elev=elev, azim=azim)
        if hasattr(ax, 'set_proj_type'):
            ax.set_proj_type('persp')
        dist = getattr(args, 'view_dist', None)
        if dist is not None:
            ax.dist = dist
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect((max_x - min_x, max_y - min_y, max_z - min_z))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def _plot_episode_sequence(episode: EpisodeTrace, maze_meta: Optional[MazeMeta], args,
                            out_path: Path, tag_label: str) -> None:
    is_3d = episode.positions.shape[1] >= 3
    if is_3d:
        fig = plt.figure(figsize=(args.figure_size, args.figure_size))
        ax = fig.add_subplot(111, projection='3d')
        _draw_maze_background_3d(ax, maze_meta, args.floor_alpha)
        path = episode.positions[:, :3]
        if path.shape[0] >= 2:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color=args.path_color,
                    linewidth=args.path_width, alpha=0.85, label='Controller trace')
    else:
        fig, ax = plt.subplots(figsize=(args.figure_size, args.figure_size))
        _draw_maze_background_2d(ax, maze_meta, args.floor_alpha)
        path = episode.positions[:, :2]
        if path.shape[0] >= 2:
            ax.plot(path[:, 0], path[:, 1], color=args.path_color, linewidth=args.path_width,
                    alpha=0.85, label='Controller trace')

    dispersions = [segment.dispersion for segment in episode.segments if segment.dispersion is not None]
    colors: List[Tuple[float, float, float, float]] = []
    norm: Optional[Normalize] = None
    if dispersions:
        values = np.asarray(dispersions, dtype=np.float32)
        vmin = float(values.min())
        vmax = float(values.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(args.sequence_cmap)
        colors = [cmap(norm(val)) for val in values]
    else:
        colors = [args.default_marker_color] * len(episode.segments)

    landing_alpha = getattr(args, 'landing_alpha', 0.45)
    landing_size = getattr(args, 'landing_marker_size', 32.0)

    for idx, segment in enumerate(episode.segments):
        color = colors[idx] if idx < len(colors) else args.default_marker_color
        landing = _landing_point(segment)
        if is_3d:
            goal = segment.abs_goal
            ax.scatter(goal[0], goal[1], goal[2] if goal.shape[0] >= 3 else 0.0,
                       c=[color], s=args.subgoal_marker_size, marker='o', edgecolor='k',
                       linewidth=0.5, depthshade=False)
            if landing is not None:
                ax.scatter(landing[0], landing[1], landing[2] if landing.shape[0] >= 3 else 0.0,
                           c=[_with_alpha(color, landing_alpha)], s=landing_size, marker='o',
                           edgecolor='none', depthshade=False, label=None)
                _draw_landing_radius(ax, goal, landing, color, True, args)
            if args.annotate_height and goal.shape[0] >= 3:
                ax.text(goal[0], goal[1], goal[2], f'z={goal[2]:.1f}', fontsize=8,
                        ha='left', va='bottom', color='k')
        else:
            ax.scatter(segment.abs_goal[0], segment.abs_goal[1], c=[color], s=args.subgoal_marker_size,
                       marker='o', edgecolor='k', linewidth=0.5)
            if landing is not None:
                ax.scatter(landing[0], landing[1], c=[_with_alpha(color, landing_alpha)],
                           s=landing_size, marker='o', edgecolors='none')
                _draw_landing_radius(ax, segment.abs_goal, landing, color, False, args)
            if args.annotate_height and segment.abs_goal.shape[0] >= 3:
                ax.text(segment.abs_goal[0], segment.abs_goal[1], f'z={segment.abs_goal[2]:.1f}',
                        fontsize=8, ha='left', va='bottom', color='k')

    if dispersions and norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=args.sequence_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Dispersion score')

    if is_3d:
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = _axis_limits_3d(maze_meta, episode.positions[:, :3])
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(min_z, max_z)
    else:
        min_x, max_x, min_y, max_y = _axis_limits(maze_meta, path)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_title(f'{args.env_name} – episode {episode.episode_index} ({tag_label})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if is_3d:
        ax.set_zlabel('z')
        ax.view_init(elev=getattr(args, 'view_elev', 40.0), azim=getattr(args, 'view_azim', -90.0))
        if hasattr(ax, 'set_proj_type'):
            ax.set_proj_type('persp')
        dist = getattr(args, 'view_dist', None)
        if dist is not None:
            ax.dist = dist
        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect((max_x - min_x, max_y - min_y, max_z - min_z))

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def visualize_s3(args) -> None:
    checkpoints = _parse_checkpoint_list(getattr(args, 'checkpoints', None))
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    reference_starts: Optional[List[Optional[np.ndarray]]] = None
    reference_goals: Optional[List[Optional[np.ndarray]]] = None

    for tag in checkpoints:
        env, state_dim, goal_dim, controller_goal_dim, max_action, action_dim, man_scale, no_xy, maze_meta = _create_env(args)
        manager_policy, controller_policy, device = _init_policies(state_dim, goal_dim, controller_goal_dim,
                                                                   action_dim, max_action, man_scale, no_xy, args)
        model_dir = Path(args.model_dir)
        _load_checkpoint(manager_policy, controller_policy, model_dir, args.env_name, args.algo, tag, device)
        reach_net = _maybe_load_reach_net(args, controller_goal_dim, device, tag)

        episodes: List[EpisodeTrace] = []
        for ep_idx in range(args.episodes):
            start_xy = reference_starts[ep_idx] if reference_starts and ep_idx < len(reference_starts) else None
            desired_goal = reference_goals[ep_idx] if reference_goals and ep_idx < len(reference_goals) else None
            trace = _collect_episode(
                env=env,
                manager_policy=manager_policy,
                controller_policy=controller_policy,
                controller_goal_dim=controller_goal_dim,
                goal_dim=goal_dim,
                args=args,
                reach_net=reach_net,
                device=device,
                tag=tag,
                episode_index=ep_idx,
                start_xy=start_xy,
                desired_goal=desired_goal,
            )
            episodes.append(trace)

        if reference_starts is None:
            reference_starts = [episode.start_xy for episode in episodes]
            reference_goals = [episode.desired_goal for episode in episodes]

        tag_label = tag or 'base'
        overlay_path = plots_dir / f'{args.env_name}_{args.algo}_{tag_label}_overlay.png'
        _plot_checkpoint_overlay(episodes, maze_meta, args, overlay_path, tag_label)
        results[tag_label] = {'episodes': episodes, 'maze': maze_meta}

    sequence_tag = args.sequence_checkpoint or (checkpoints[-1] if checkpoints else None)
    if sequence_tag is not None:
        seq_label = sequence_tag or 'base'
        payload = results.get(seq_label)
        if payload:
            episodes = payload['episodes']
            if episodes:
                index = min(args.sequence_index, len(episodes) - 1)
                sequence_path = plots_dir / f'{args.env_name}_{args.algo}_{seq_label}_episode{index}_sequence.png'
                _plot_episode_sequence(episodes[index], payload['maze'], args, sequence_path, seq_label)
