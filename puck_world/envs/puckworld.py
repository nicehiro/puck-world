import math
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from puck_world.envs.agent import Agent
from puck_world.envs.agent_type import AgentType


class PuckWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.width = 600
        self.height = 600
        self.len_unit = 1.0
        self.vel = 20.0
        self.runner_vel = 50.0
        self.agents = {agent_type: None for agent_type in AgentType}
        self.time = 0
        self.rewards = []
        self.viewer = None
        self.radian2angle = math.pi / 180

        self.np_random = None
        self.action_space_bound = (-180, 180)
        self.action_space = spaces.Box(np.array(self.action_space_bound[0]),
                                       np.array(self.action_space_bound[1]))

        self.normal_reward = 1
        self.special_reward = 10

        self.state = [0, 0, 0, 0]

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_agent(self, agent):
        if agent.type == AgentType.Chaser:
            self.agents[AgentType.Chaser] = agent
        else:
            self.agents[AgentType.Runner] = agent

    def __dis(self, s_x, s_y, t_x, t_y):
        return math.sqrt((s_x - t_x) ** 2 + (s_y - t_y) ** 2)

    def is_done(self):
        chaser = self.agents[AgentType.Chaser]
        runner = self.agents[AgentType.Runner]
        return self.__dis(chaser.x, chaser.y, runner.x, runner.y) <= min(chaser.r, runner.r)

    def step(self, agent_type, action:int):
        """
        Agent how to move based on action in this envs.

        :param name: agent's name
        :param action: agent's action
        :return: next_state, reward, done, info
        """
        assert self.action_space_bound[0] <= action <= self.action_space_bound[1], \
            '%r (%s) invalid' % (action, type(action))
        agent = self.agents[agent_type]
        if agent_type == AgentType.Chaser:
            target = self.agents[AgentType.Runner]
        else:
            target = self.agents[AgentType.Chaser]
        delta_x = self.vel * math.cos(action * self.radian2angle)
        delta_y = self.vel * math.sin(action * self.radian2angle)
        new_x, new_y = agent.x + delta_x, agent.y + delta_y

        if new_x - agent.r <= 0:
            new_x = agent.r
        if new_x + agent.r >= self.width:
            new_x = self.width - agent.r
        if new_y - agent.r <= 0:
            new_y = agent.r
        if new_y + agent.r >= self.height:
            new_y = self.height - agent.r

        reward_prefix = 1
        reward = self.normal_reward
        done = self.is_done()
        if agent.type == AgentType.Chaser:
            if not done:
                reward_prefix = -1
            else:
                reward = self.special_reward
        if agent.type == AgentType.Runner:
            if done:
                reward_prefix = -1
                reward = self.special_reward
        reward = reward_prefix * reward
        info = {}
        self.state = [new_x, new_y, target.x, target.y]
        agent.x, agent.y = new_x, new_y
        return self.state, reward, done, info

    def reset(self):
        chaser_x = self.width * random.random()
        chaser_y = self.height * random.random()
        runner_x = self.width * random.random()
        runner_y = self.height * random.random()
        chaser = self.agents[AgentType.Chaser]
        runner = self.agents[AgentType.Runner]
        chaser.x, chaser.y, runner.x, runner.y = chaser_x, chaser_y, runner_x, runner_y
        self.state = [chaser_x, chaser_y, runner_x, runner_y]
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        chaser = self.agents[AgentType.Chaser]
        runner = self.agents[AgentType.Runner]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            chaser_obj = rendering.make_circle(chaser.r, 30, True)
            chaser_obj.set_color(*chaser.color)
            self.viewer.add_geom(chaser_obj)
            self.chaser_trans = rendering.Transform()
            chaser_obj.add_attr(self.chaser_trans)

            runner_obj = rendering.make_circle(runner.r, 30, True)
            runner_obj.set_color(*runner.color)
            self.viewer.add_geom(runner_obj)
            self.runner_trans = rendering.Transform()
            runner_obj.add_attr(self.runner_trans)

        self.chaser_trans.set_translation(chaser.x, chaser.y)
        self.runner_trans.set_translation(runner.x, runner.y)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = PuckWorld()
    chaser = Agent(50, 50, 30, (1, 0, 0), AgentType.Chaser)
    runner = Agent(500, 500, 30, (0, 0, 1), AgentType.Runner)

    env.add_agent(chaser)
    env.add_agent(runner)

    env.reset()

    for i in range(10000):
        env.render()
        if i % 2 == 0:
            agent_type = AgentType.Chaser
        else:
            agent_type = AgentType.Runner
        env.step(agent_type, env.action_space.sample())
