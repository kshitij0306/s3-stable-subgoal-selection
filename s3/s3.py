import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

import numpy as np

from s3.models import ControllerActor, ControllerCritic, \
    ManagerActor, ManagerCritic
from s3.dispersion import dispersion_score, dispersion_score_batch


"""
HIRO part adapted from
https://github.com/bhairavmehta95/data-efficient-hrl/blob/master/hiro/hiro.py
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def var(tensor):
    return tensor.to(device)


def get_tensor(z):
    if z is None:
        return None

    if isinstance(z, torch.Tensor):
        tensor = z.to(device=device, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    z_arr = np.asarray(z)
    if z_arr.dtype == np.dtype("O"):
        return None

    tensor = torch.as_tensor(z_arr, dtype=torch.float32, device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _stack_goals_for_reach(states_np: np.ndarray, goals_np: np.ndarray, goal_dim: int) -> np.ndarray:
    """Construct batched reach-net inputs from states and goals."""

    goals_np = np.asarray(goals_np)
    if goals_np.ndim == 1:
        goals_np = goals_np.reshape(1, -1)

    num_entries = goals_np.shape[0]
    if goal_dim == 0:
        return np.zeros((num_entries, 0), dtype=np.float32)

    states_np = np.asarray(states_np)
    if states_np.ndim == 1:
        state_prefix = np.tile(states_np[:goal_dim], (num_entries, 1))
    else:
        state_prefix = states_np[:, :goal_dim]
        if state_prefix.shape[0] != num_entries:
            if state_prefix.shape[0] == 1:
                state_prefix = np.tile(state_prefix, (num_entries, 1))
            else:
                repeat_factor = num_entries // state_prefix.shape[0]
                state_prefix = np.repeat(state_prefix, repeat_factor, axis=0)

    goal_prefix = goals_np[:, :goal_dim].astype(np.float32)
    return np.hstack([state_prefix.astype(np.float32), goal_prefix])


def compute_phi(manager_policy, reach_net, state_np, goal_ctx_np,
                goal_dim, device, args, samples: int = 1) -> float:
    """Estimate Φ(s) via dispersion-based potential."""

    if reach_net is None:
        return 0.0

    if samples <= 0:
        return 0.0

    goals = []
    for _ in range(samples):
        g = manager_policy.sample_goal(state_np, goal_ctx_np)[:goal_dim]
        goals.append(g.astype(np.float32))

    if not goals:
        return 0.0

    goal_stack = np.stack(goals)
    inp_np = _stack_goals_for_reach(state_np, goal_stack, goal_dim)

    with torch.no_grad():
        inp = torch.from_numpy(inp_np).to(device)
        pi, mu, logS = reach_net(inp)

        if args.dispersion in ("chance", "cvar"):
            scores = []
            for idx in range(len(goals)):
                score, _ = dispersion_score(
                    pi[idx:idx+1], mu[idx:idx+1], logS[idx:idx+1],
                    state_np=state_np,
                    subgoal_np=goals[idx],
                    d=goal_dim,
                    args=args,
                    device=device,
                    collect_stats=False,
                )
                scores.append(float(score))
            score_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
        else:
            score_tensor = dispersion_score_batch(
                pi, mu, logS,
                state_np=state_np,
                subgoal_np=goal_stack,
                d=goal_dim,
                args=args,
                device=device,
            )

    return -float(score_tensor.mean().item())


def compute_phi_batch(manager_policy, reach_net, states_np, goal_ctx_np,
                      goal_dim, device, args, samples: int = 1) -> np.ndarray:
    """Batched Φ computation for a batch of states."""

    if reach_net is None or samples <= 0 or goal_dim == 0:
        batch_size = np.asarray(states_np).shape[0]
        return np.zeros(batch_size, dtype=np.float32)

    states_np = np.asarray(states_np)
    goal_ctx_np = np.asarray(goal_ctx_np)
    batch_size = states_np.shape[0]

    sampled_goals = manager_policy.sample_goal_batch(
        states_np, goal_ctx_np, repeats=samples, to_numpy=True
    )  # (B, S, action_dim)

    if sampled_goals is None:
        return np.zeros(batch_size, dtype=np.float32)

    sampled_goals = sampled_goals[:, :samples, :]
    flat_states = np.repeat(states_np[:, None, :], samples, axis=1).reshape(batch_size * samples, -1)
    flat_goals = sampled_goals.reshape(batch_size * samples, -1)
    inp_np = _stack_goals_for_reach(flat_states, flat_goals[:, :goal_dim], goal_dim)

    with torch.no_grad():
        inp = torch.from_numpy(inp_np).float().to(device)
        pi, mu, logS = reach_net(inp)

        if args.dispersion in ("chance", "cvar"):
            scores = []
            for idx in range(pi.shape[0]):
                score, _ = dispersion_score(
                    pi[idx:idx+1], mu[idx:idx+1], logS[idx:idx+1],
                    state_np=flat_states[idx],
                    subgoal_np=flat_goals[idx],
                    d=goal_dim,
                    args=args,
                    device=device,
                    collect_stats=False,
                )
                scores.append(float(score))
            score_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
        else:
            score_tensor = dispersion_score_batch(
                pi, mu, logS,
                state_np=flat_states,
                subgoal_np=flat_goals,
                d=goal_dim,
                args=args,
                device=device,
            )

    score_tensor = score_tensor.view(batch_size, samples)
    return -score_tensor.mean(dim=1).cpu().numpy().astype(np.float32)


class Manager(object):
    def __init__(self, state_dim, goal_dim, action_dim, actor_lr,
                 critic_lr, candidate_goals, correction=True,
                 scale=10, actions_norm_reg=0, policy_noise=0.2,
                 noise_clip=0.5, goal_loss_coeff=0, absolute_goal=False):
        self.scale = scale
        self.actor = ManagerActor(state_dim, goal_dim, action_dim,
                                  scale=scale, absolute_goal=absolute_goal).to(device)
        self.actor_target = ManagerActor(state_dim, goal_dim, action_dim,
                                         scale=scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = ManagerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ManagerCritic(state_dim, goal_dim, action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr, weight_decay=0.0001)

        self.amp_enabled = torch.cuda.is_available()
        self.critic_scaler = GradScaler(enabled=self.amp_enabled)
        self.actor_scaler = GradScaler(enabled=self.amp_enabled)

        self.action_norm_reg = 0

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.candidate_goals = candidate_goals
        self.correction = correction
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.goal_loss_coeff = goal_loss_coeff
        self.absolute_goal = absolute_goal
        self.latest_phi_stats = None

    def set_eval(self):
        self.actor.set_eval()
        self.actor_target.set_eval()

    def set_train(self):
        self.actor.set_train()
        self.actor_target.set_train()

    def sample_goal(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)

        with torch.no_grad():
            action = self.actor(state, goal)

        if to_numpy:
            return action.cpu().numpy().squeeze()
        else:
            return action.squeeze()

    def sample_goal_batch(self, states, goals, repeats: int = 1, to_numpy: bool = True):
        states_tensor = get_tensor(states)
        goals_tensor = get_tensor(goals)

        if states_tensor is None or goals_tensor is None:
            return None

        batch_size = states_tensor.shape[0]
        if repeats > 1:
            states_tensor = states_tensor.repeat_interleave(repeats, dim=0)
            goals_tensor = goals_tensor.repeat_interleave(repeats, dim=0)

        with torch.no_grad():
            actions = self.actor(states_tensor, goals_tensor)

        if repeats > 1:
            actions = actions.view(batch_size, repeats, -1)
        else:
            actions = actions.unsqueeze(1)

        if to_numpy:
            return actions.cpu().numpy()
        return actions

    def value_estimate(self, state, goal, subgoal):
        return self.critic(state, goal, subgoal)

    def actor_loss(self, state, goal, a_net, r_margin):
        actions = self.actor(state, goal)
        q_pred = self.critic.Q1(state, goal, actions)
        eval = -q_pred.mean()
        # eval = -self.critic.Q1(state, goal, actions).mean()
        norm = torch.norm(actions)*self.action_norm_reg
        if a_net is None:
            return eval + norm
        else:
            goal_loss = torch.clamp(F.pairwise_distance(
                a_net(state[:, :self.action_dim]), a_net(state[:, :self.action_dim] + actions)) - r_margin, min=0.).mean()
            return eval + norm, goal_loss

    def off_policy_corrections(self, controller_policy, batch_size, subgoals, x_seq, a_seq):
        first_x = [x[0] for x in x_seq]
        last_x = [x[-1] for x in x_seq]

        # Shape: (batchsz, 1, subgoal_dim)
        diff_goal = (np.array(last_x) - np.array(first_x))[:, np.newaxis, :self.action_dim]

        # Shape: (batchsz, 1, subgoal_dim)
        original_goal = np.array(subgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.scale[None, None, :self.action_dim],
                                        size=(batch_size, self.candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.scale[:self.action_dim], self.scale[:self.action_dim])

        # Shape: (batchsz, 10, subgoal_dim)
        candidates = np.concatenate([original_goal, diff_goal, random_goals], axis=1)
        # print("DEBUG Candidates shape: ", candidates.shape)
        # print("DEBUG candidates: ", candidates)
        # print(np.array(x_seq).shape)
        x_seq = np.array(x_seq)[:, :-1, :]
        a_seq = np.array(a_seq)
        seq_len = len(x_seq[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = a_seq[0][0].shape
        obs_dim = x_seq[0][0].shape
        ncands = candidates.shape[1]

        true_actions = a_seq.reshape((new_batch_sz,) + action_dim)
        observations = x_seq.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            candidate = controller_policy.multi_subgoal_transition(x_seq, candidates[:, c])
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = controller_policy.select_action(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, controller_policy, replay_buffer, iterations, batch_size=100, discount=0.99,
              tau=0.005, a_net=None, r_margin=None, *, reach_net=None, phi_device=None,
              args=None, goal_dim=None, phi_samples: int = 4, intrinsic_scale: float = 1.0,
              intrinsic_mix: float = 1.0):
        avg_act_loss, avg_crit_loss = 0., 0.
        if a_net is not None:
            avg_goal_loss = 0.
        self.latest_phi_stats = None
        for it in range(iterations):
            # Sample replay buffer
            x, y, g, sgorig, r, d, xobs_seq, a_seq = replay_buffer.sample(batch_size)
            batch_size = min(batch_size, x.shape[0])

            if self.correction and not self.absolute_goal:
                sg = self.off_policy_corrections(controller_policy, batch_size,
                                                 sgorig, xobs_seq, a_seq)
            else:
                sg = sgorig

            state = get_tensor(x)
            next_state = get_tensor(y)
            # print(g)
            goal = get_tensor(g)
            subgoal = get_tensor(sg)

            shaped_reward = r
            torch_device = phi_device if phi_device is not None else device
            if reach_net is not None and args is not None and goal_dim is not None and torch_device is not None:
                phi_start_arr = compute_phi_batch(
                    self,
                    reach_net,
                    states_np=x,
                    goal_ctx_np=g,
                    goal_dim=goal_dim,
                    device=torch_device,
                    args=args,
                    samples=phi_samples,
                )
                phi_end_arr = compute_phi_batch(
                    self,
                    reach_net,
                    states_np=y,
                    goal_ctx_np=g,
                    goal_dim=goal_dim,
                    device=torch_device,
                    args=args,
                    samples=phi_samples,
                )
                shaping_delta = discount * phi_end_arr - phi_start_arr
                mix_coeff = max(0.0, float(intrinsic_mix))
                shaped_reward = r + (mix_coeff * intrinsic_scale * shaping_delta)[:, None]
                if phi_start_arr.size > 0:
                    self.latest_phi_stats = {
                        "phi_start_mean": float(phi_start_arr.mean()),
                        "phi_end_mean": float(phi_end_arr.mean()),
                        "pbrs_mean": float((mix_coeff * intrinsic_scale * shaping_delta).mean()),
                    }

            reward = get_tensor(shaped_reward)
            done = get_tensor(1 - d)

            noise = torch.randn_like(subgoal, device=device) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            with autocast(enabled=self.amp_enabled):
                target_action = self.actor_target(next_state, goal)
                noise_cast = noise.to(target_action.dtype)
                scale = self.actor.scale.to(target_action.dtype)
                raw_action = target_action + noise_cast
                next_action = torch.minimum(raw_action, scale)
                next_action = torch.maximum(next_action, -scale)
                target_Q1, target_Q2 = self.critic_target(next_state, goal, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            with autocast(enabled=self.amp_enabled):
                current_Q1, current_Q2 = self.value_estimate(state, goal, subgoal)
                critic_loss = (
                    self.criterion(current_Q1, target_Q_no_grad) +
                    self.criterion(current_Q2, target_Q_no_grad)
                )

            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()

            with autocast(enabled=self.amp_enabled):
                goal_loss = None
                if a_net is None:
                    total_actor_loss = self.actor_loss(state, goal, a_net, r_margin)
                else:
                    actor_loss_core, goal_loss = self.actor_loss(state, goal, a_net, r_margin)
                    total_actor_loss = actor_loss_core + self.goal_loss_coeff * goal_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.actor_scaler.scale(total_actor_loss).backward()
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()

            avg_act_loss += float(total_actor_loss.detach().item())
            avg_crit_loss += float(critic_loss.detach().item())
            if a_net is not None and goal_loss is not None:
                avg_goal_loss += float(goal_loss.detach().item())

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if a_net is None:
            return avg_act_loss / iterations, avg_crit_loss / iterations
        else:
            return avg_act_loss / iterations, avg_crit_loss / iterations, avg_goal_loss / iterations

    def load_pretrained_weights(self, filename):
        state = torch.load(filename)
        self.actor.encoder.load_state_dict(state)
        self.actor_target.encoder.load_state_dict(state)
        print("Successfully loaded Manager encoder.")

    def save(self, dir, env_name, algo):
        torch.save(self.actor.state_dict(), "{}/{}_{}_ManagerActor.pth".format(dir, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}_{}_ManagerCritic.pth".format(dir, env_name, algo))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_ManagerActorTarget.pth".format(dir, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_ManagerCriticTarget.pth".format(dir, env_name, algo))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, suffix=None):
        suffix_part = f"_{suffix}" if suffix else ""
        base_path = os.path.join(dir, f"{env_name}_{algo}")
        map_target = device

        self.actor.load_state_dict(torch.load(f"{base_path}_ManagerActor{suffix_part}.pth", map_location=map_target))
        self.critic.load_state_dict(torch.load(f"{base_path}_ManagerCritic{suffix_part}.pth", map_location=map_target))
        self.actor_target.load_state_dict(torch.load(f"{base_path}_ManagerActorTarget{suffix_part}.pth", map_location=map_target))
        self.critic_target.load_state_dict(torch.load(f"{base_path}_ManagerCriticTarget{suffix_part}.pth", map_location=map_target))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ManagerCriticOptim.pth".format(dir, env_name, algo)))



class Controller(object):
    def __init__(self, state_dim, goal_dim, action_dim, max_action, actor_lr,
                 critic_lr, repr_dim=15, no_xy=True, policy_noise=0.2, noise_clip=0.5,
                 absolute_goal=False
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.no_xy = no_xy
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.absolute_goal = absolute_goal
        self.criterion = nn.SmoothL1Loss()

        self.actor = ControllerActor(state_dim, goal_dim, action_dim,
                                     scale=max_action).to(device)
        self.actor_target = ControllerActor(state_dim, goal_dim, action_dim,
                                            scale=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
            lr=actor_lr)

        self.critic = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target = ControllerCritic(state_dim, goal_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
            lr=critic_lr, weight_decay=0.0001)

        self.amp_enabled = torch.cuda.is_available()
        self.critic_scaler = GradScaler(enabled=self.amp_enabled)
        self.actor_scaler = GradScaler(enabled=self.amp_enabled)

    def clean_obs(self, state, dims=2):
        if self.no_xy:
            with torch.no_grad():
                mask = torch.ones_like(state)
                if len(state.shape) == 3:
                    mask[:, :, :dims] = 0
                elif len(state.shape) == 2:
                    mask[:, :dims] = 0
                elif len(state.shape) == 1:
                    mask[:dims] = 0

                return state*mask
        else:
            return state

    def select_action(self, state, sg, evaluation=False):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        with torch.no_grad():
            action = self.actor(state, sg)
        return action.cpu().numpy().squeeze()

    def value_estimate(self, state, sg, action):
        state = self.clean_obs(get_tensor(state))
        sg = get_tensor(sg)
        action = get_tensor(action)
        return self.critic(state, sg, action)

    def actor_loss(self, state, sg):
        # return -self.critic.Q1(state, sg, self.actor(state, sg)).mean()
        q_pred = self.critic.Q1(state, sg, self.actor(state, sg))
        return -q_pred.mean()

    def subgoal_transition(self, state, subgoal, next_state):
        if self.absolute_goal:
            return subgoal
        else:
            if len(state.shape) == 1:  # check if batched
                return state[:self.goal_dim] + subgoal - next_state[:self.goal_dim]
            else:
                return state[:, :self.goal_dim] + subgoal -\
                       next_state[:, :self.goal_dim]

    def multi_subgoal_transition(self, states, subgoal):
        subgoals = (subgoal + states[:, 0, :self.goal_dim])[:, None] - \
                   states[:, :, :self.goal_dim]
        return subgoals

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):
        avg_act_loss, avg_crit_loss = 0., 0.
        for it in range(iterations):
            x, y, sg, u, r, d, _, _ = replay_buffer.sample(batch_size)
            next_g = get_tensor(self.subgoal_transition(x, sg, y))
            state = self.clean_obs(get_tensor(x))
            action = get_tensor(u)
            sg = get_tensor(sg)
            done = get_tensor(1 - d)
            reward = get_tensor(r)
            next_state = self.clean_obs(get_tensor(y))

            noise = torch.randn_like(action, device=device) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            with autocast(enabled=self.amp_enabled):
                target_action = self.actor_target(next_state, next_g)
                noise_cast = noise.to(target_action.dtype)
                scale = self.actor.scale.to(target_action.dtype)
                raw_action = target_action + noise_cast
                next_action = torch.minimum(raw_action, scale)
                next_action = torch.maximum(next_action, -scale)
                target_Q1, target_Q2 = self.critic_target(next_state, next_g, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q)
            target_Q_no_grad = target_Q.detach()

            with autocast(enabled=self.amp_enabled):
                current_Q1, current_Q2 = self.critic(state, sg, action)
                critic_loss = (
                    self.criterion(current_Q1, target_Q_no_grad) +
                    self.criterion(current_Q2, target_Q_no_grad)
                )

            self.critic_optimizer.zero_grad(set_to_none=True)
            self.critic_scaler.scale(critic_loss).backward()
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()

            with autocast(enabled=self.amp_enabled):
                actor_loss = self.actor_loss(state, sg)

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()

            avg_act_loss += float(actor_loss.detach().item())
            avg_crit_loss += float(critic_loss.detach().item())

            # Update the target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return avg_act_loss / iterations, avg_crit_loss / iterations

    def save(self, dir, env_name, algo):
        torch.save(self.actor.state_dict(), "{}/{}_{}_ControllerActor.pth".format(dir, env_name, algo))
        torch.save(self.critic.state_dict(), "{}/{}_{}_ControllerCritic.pth".format(dir, env_name, algo))
        torch.save(self.actor_target.state_dict(), "{}/{}_{}_ControllerActorTarget.pth".format(dir, env_name, algo))
        torch.save(self.critic_target.state_dict(), "{}/{}_{}_ControllerCriticTarget.pth".format(dir, env_name, algo))
        # torch.save(self.actor_optimizer.state_dict(), "{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo))
        # torch.save(self.critic_optimizer.state_dict(), "{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo))

    def load(self, dir, env_name, algo, suffix=None):
        suffix_part = f"_{suffix}" if suffix else ""
        base_path = os.path.join(dir, f"{env_name}_{algo}")
        map_target = device

        self.actor.load_state_dict(torch.load(f"{base_path}_ControllerActor{suffix_part}.pth", map_location=map_target))
        self.critic.load_state_dict(torch.load(f"{base_path}_ControllerCritic{suffix_part}.pth", map_location=map_target))
        self.actor_target.load_state_dict(torch.load(f"{base_path}_ControllerActorTarget{suffix_part}.pth", map_location=map_target))
        self.critic_target.load_state_dict(torch.load(f"{base_path}_ControllerCriticTarget{suffix_part}.pth", map_location=map_target))
        # self.actor_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerActorOptim.pth".format(dir, env_name, algo)))
        # self.critic_optimizer.load_state_dict(torch.load("{}/{}_{}_ControllerCriticOptim.pth".format(dir, env_name, algo)))
