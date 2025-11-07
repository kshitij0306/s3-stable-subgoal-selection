import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - handled at runtime
    gym = None


def _require_gymnasium(env_name: str):
    if gym is None:
        raise ImportError(
            f"Gymnasium is required to use '{env_name}'. "
            "Install with `pip install gymnasium[mujoco]` as per Mujoco's setup instructions."
        )


def _get_body_com(env, body: str) -> np.ndarray:
    # Gymnasium Mujoco envs expose MuJoCo structures via `unwrapped`.
    return np.array(env.unwrapped.get_body_com(body), dtype=np.float32)


def _concat_with_achieved(raw_obs: np.ndarray, achieved: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_obs, dtype=np.float32).ravel()
    return np.concatenate([achieved.astype(np.float32), raw], axis=0)


GYMNASIUM_ENV_SPECS = {
    "Reacher-v5": {
        "controller_goal_dim": 2,
        "goal_dim": 2,
        "manager_action_scale": np.array([0.3, 0.3], dtype=np.float32),
        "distance_threshold": 0.03,
        "achieved_extractor": lambda env: _get_body_com(env, "fingertip")[:2],
        "desired_extractor": lambda env: np.array(env.unwrapped.goal[:2], dtype=np.float32),
        "obs_adapter": _concat_with_achieved,
    },
    "Pusher-v5": {
        "controller_goal_dim": 2,
        "goal_dim": 2,
        "manager_action_scale": np.array([0.4, 0.4], dtype=np.float32),
        "distance_threshold": 0.07,
        "achieved_extractor": lambda env: _get_body_com(env, "object")[:2],
        "desired_extractor": lambda env: _get_body_com(env, "goal")[:2],
        "obs_adapter": _concat_with_achieved,
    },
}


class GymnasiumGoalEnv:
    """Light-weight wrapper that converts Gymnasium Mujoco tasks into goal-conditioned dict envs."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_name: str, seed: int | None = None, **make_kwargs):
        if env_name not in GYMNASIUM_ENV_SPECS:
            raise ValueError(f"Unsupported Gymnasium env '{env_name}'.")

        _require_gymnasium(env_name)
        self.env_name = env_name
        make_kwargs = dict(make_kwargs)
        render_mode = make_kwargs.pop("render_mode", None)
        if render_mode is not None:
            self.env = gym.make(env_name, render_mode=render_mode, **make_kwargs)
        else:
            self.env = gym.make(env_name, **make_kwargs)
        self.evaluate = False
        self.count = 0
        self.render_mode = render_mode

        spec = GYMNASIUM_ENV_SPECS[env_name]
        self.controller_goal_dim = int(spec["controller_goal_dim"])
        self.goal_dim = int(spec.get("goal_dim", self.controller_goal_dim))
        self.manager_action_scale = np.asarray(
            spec.get("manager_action_scale", np.ones(self.controller_goal_dim, dtype=np.float32)),
            dtype=np.float32,
        )
        self.distance_threshold = float(spec.get("distance_threshold", 0.05))
        self._achieved_extractor = spec["achieved_extractor"]
        self._desired_extractor = spec["desired_extractor"]
        self._obs_adapter = spec.get("obs_adapter", _concat_with_achieved)
        self._reward_from_distance = spec.get(
            "reward_fn",
            lambda distance, env_reward: -float(distance),
        )
        self._last_distance = np.inf

        env_steps = getattr(getattr(self.env, "spec", None), "max_episode_steps", None)
        self.max_episode_steps = int(spec.get("max_episode_steps", env_steps or 200))

        self.success_fn = lambda reward: self._last_distance <= self.distance_threshold

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int | None = None):
        if seed is None:
            return
        # Gymnasium's recommended seeding protocol
        self.env.reset(seed=seed)
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        if hasattr(self.env.observation_space, "seed"):
            self.env.observation_space.seed(seed)

    def reset(self):
        self.count = 0
        obs, _info = self.env.reset()
        return self._build_transition(obs)

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.count += 1
        wrapped_obs = self._build_transition(obs)
        reward = self._reward_from_distance(self._last_distance, env_reward)
        done = terminated or truncated or self.count >= self.max_episode_steps
        info = dict(info)
        info.setdefault("env_reward", env_reward)
        info.setdefault("distance", self._last_distance)
        return wrapped_obs, reward, done, info

    def render(self, mode: str | None = None):
        target_mode = mode or getattr(self.env, "render_mode", None)
        if target_mode not in (None, "human", "rgb_array"):
            raise NotImplementedError(f"Render mode '{target_mode}' not supported.")

        if mode is not None and getattr(self.env, "render_mode", None) != mode:
            if hasattr(self.env, "render_mode"):
                self.env.render_mode = mode
                self.render_mode = mode
            else:  # fallback for legacy signature
                try:
                    return self.env.render(mode=mode)
                except TypeError as exc:
                    raise NotImplementedError(f"Render mode '{mode}' not supported.") from exc

        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def spec(self):
        return getattr(self.env, "spec", None)

    def _build_transition(self, raw_obs):
        achieved = np.asarray(self._achieved_extractor(self.env), dtype=np.float32).copy()
        desired = np.asarray(self._desired_extractor(self.env), dtype=np.float32).copy()
        distance = np.linalg.norm(achieved[: self.goal_dim] - desired[: self.goal_dim])
        self._last_distance = float(distance)

        state = self._obs_adapter(np.asarray(raw_obs, dtype=np.float32), achieved)

        return {
            "observation": state.astype(np.float32),
            "achieved_goal": achieved[: self.controller_goal_dim].astype(np.float32),
            "desired_goal": desired[: self.goal_dim].astype(np.float32),
        }
