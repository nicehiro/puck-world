import math
import random
from enum import unique, Enum

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Entity():
    def __init__(self):
        self.type = None
        self.name = None
        self.state = None
        self.movable = False
        self.radius = None
        self.color = None


@unique
class EntityType(Enum):
    Agent = 0
    Landmark = 1


class Agent(Entity):
    def __init__(self, name='agent'):
        self.type = EntityType.Agent
        self.name = name
        # state info: [agent_x, agent_y]
        self.state = [150, 150]
        # vector info: [vector_x, vector_y]
        self.vector = [0, 0]
        self.vector_threshold = [-5, 5]
        self.accelerate = 2
        self.movable = True
        self.radius = 0.3
        self.unit = 100
        self.color = (1.0, 0.0, 0.0)

    def update_vector(self, radian: float):
        self.vector[0] += self.accelerate * math.cos(radian)
        self.vector[1] += self.accelerate * math.sin(radian)
        for i in range(len(self.vector)):
            if self.vector[i] > self.vector_threshold[1]:
                self.vector[i] = self.vector_threshold[1]
            if self.vector[i] < self.vector_threshold[0]:
                self.vector[i] = self.vector_threshold[0]

    def update_state(self):
        self.state[0] += self.vector[0] / self.unit
        self.state[1] += self.vector[1] / self.unit


class Landmark(Entity):
    def __init__(self, name='landmark'):
        self.type = EntityType.Landmark
        self.name = name
        # state info: [mark_x, mark_y]
        self.state = [0, 0]
        self.movable = False
        self.radius = 0.1
        self.unit = 100
        self.color = (0.0, 1.0, 0.0)


class PuckWorld(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.width = 3
        self.height = 3
        self.unit = 100
        self.time = 0
        self.rewards = []
        self.viewer = None
        self.np_random = None
        # action_space
        # (-pi, pi)
        self.action_space = spaces.Box(low=np.array([-math.pi]),
                                       high=np.array([math.pi]))
        # observation space
        # (agent_x, agent_y, agent_v_x, agent_v_y, mark_x, mark_y)
        self.observation_space = spaces.Box(low=np.array([0, 0, -5, -5, 0, 0]),
                                            high=np.array([3, 3, 5, 5, 3, 3]),
                                            dtype=np.float)
        self.agent = Agent()
        self.landmark = Landmark()
        self.state = None
        # self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def __dis(self, s_x, s_y, t_x, t_y):
        return math.sqrt((s_x - t_x) ** 2 + (s_y - t_y) ** 2)

    def __is_done(self):
        return self.__dis(*self.agent.state, *self.landmark.state) \
               <= max(self.agent.radius, self.landmark.radius)

    def step(self, action):
        action = action[0]
        self.agent.update_vector(action)
        self.agent.update_state()
        if self.agent.state[0] < 0:
            self.agent.state[0] = 0
        if self.agent.state[0] > self.width:
            self.agent.state[0] = self.width
        if self.agent.state[1] < 0:
            self.agent.state[1] = 0
        if self.agent.state[1] > self.height:
            self.agent.state[1] = self.height
        self.state = self.__concat_state()
        info = {}
        return self.state, self.__get_reward(), self.__is_done(), info

    def __get_reward(self):
        reward = 1 - math.exp(1e-2 * self.__dis(*self.agent.state, *self.landmark.state))
        return reward

    def __concat_state(self):
        return self.agent.state + self.agent.vector + self.landmark.state

    def reset(self):
        rand_wrap = lambda x: x * random.random()
        self.landmark.state = list(map(rand_wrap, [self.width, self.height]))
        self.state = self.__concat_state()
        return self.state

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width*self.unit, self.height*self.unit)
            landmark = rendering.make_circle(self.landmark.radius*self.unit)
            landmark.set_color(*self.landmark.color)
            self.viewer.add_geom(landmark)
            self.landmark_trans = rendering.Transform()
            landmark.add_attr(self.landmark_trans)

            agent_obj = rendering.make_circle(self.agent.radius*self.unit)
            agent_obj.set_color(*self.agent.color)
            self.viewer.add_geom(agent_obj)
            self.agent_trans = rendering.Transform()
            agent_obj.add_attr(self.agent_trans)
        self.agent_trans.set_translation(*[x * self.unit for x in self.agent.state])
        self.landmark_trans.set_translation(*[x * self.unit for x in self.landmark.state])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = PuckWorld()
    for _ in range(100):
        env.reset()
        for _ in range(100):
            env.render(close=False)
            a = env.action_space.sample()
            _, _, done, _ = env.step(a.item())
            if done:
                break
