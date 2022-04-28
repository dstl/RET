"""Runner for the AI Gym run of IADS."""

from gym_ret.envs.ret_env import RetEnv

from iads.model import IADS

env = RetEnv(IADS)

obs = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    print(rewards)
