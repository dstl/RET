"""AI Gym for ret."""

from gymnasium.envs.registration import register

register(
    id="ret-v0",
    entry_point="gym_ret.envs:RetEnv",
)
