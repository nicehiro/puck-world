import math
import random
import time

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from puck_world.envs.agent import AgentWithWheel
from puck_world.envs.agent_type import AgentType


class PuckWorldWheel(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self, fps=60):
        self.width = 400
        self.height = 400
        self.len_unit = 1.0
        self.vel = 20.0
        self.runner_vel = 50.0
        self.agents = {agent_type: None for agent_type in AgentType}
        self.time = 0
        self.rewards = []
        self.viewer = None
        self.radian2angle = math.pi / 180
        self.RAD2DEG = 57.29577951308232

        self.np_random = None
        self.action_space_bound = (-180, 180)
        self.action_space = spaces.Box(np.array(self.action_space_bound[0]),
                                       np.array(self.action_space_bound[1]))

        self.normal_reward = 1
        self.special_reward = 10

        self.state = [0, 0, 0, 0]
        self.fps = fps

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
        if agent_type == AgentType.Chaser:
            new_x, new_y, direction = self._wheel_move(agent.x, agent.y, agent.r, agent.direction, agent.left_v, agent.right_v)
            agent.direction = direction

        if new_x - agent.r <= 0:
            new_x = agent.r
        if new_x + agent.r >= self.width:
            new_x = self.width - agent.r
        if new_y - agent.r <= 0:
            new_y = agent.r
        if new_y + agent.r >= self.height:
            new_y = self.height - agent.r

        # reward_prefix = 1
        # reward = self.normal_reward
        # if agent.type == AgentType.Chaser:
        #     if not done:
        #         reward_prefix = -1
        #     else:
        #         reward = self.special_reward
        # if agent.type == AgentType.Runner:
        #     if done:
        #         reward_prefix = -1
        #         reward = self.special_reward
        # reward = reward_prefix * reward
        done = self.is_done()
        reward = self._cal_reward(agent, target)
        info = {}
        self.state = [new_x, new_y, target.x, target.y]
        if agent.type == AgentType.Chaser:
            self.state.append(agent.direction)
        agent.x, agent.y = new_x, new_y
        return self.state, reward, done, info

    def _cal_reward(self, chaser, runner):
        """
        Calculate cahser's reward regard as distance of chaser and runner.
        :param chaser: Chaser
        :param runner: Runner
        :return: reward
        """
        return - 10 * math.sqrt(((chaser.x - runner.x) ** 2 + (chaser.y - runner.y) ** 2))

    def reset(self):
        chaser_x = self.width * random.random()
        chaser_y = self.height * random.random()
        runner_x = self.width * random.random()
        runner_y = self.height * random.random()
        chaser = self.agents[AgentType.Chaser]
        runner = self.agents[AgentType.Runner]
        chaser.x, chaser.y, runner.x, runner.y = chaser_x, chaser_y, runner_x, runner_y
        self.state = [chaser_x, chaser_y, runner_x, runner_y, chaser.direction]
        return self.state

    def _wheel_move(self, x, y, r, direction, left_v, right_v):
        """
        Get chaser's next position and direction.
        :param x: chaser_x
        :param y: chaser_y
        :param direction: chaser's direction
        :param left_v: left wheel vector
        :param right_v: right wheel vector
        :return: new_x, new_y, new_direction
        """
        l1 = left_v / self.fps
        l2 = right_v / self.fps
        direct_rad = direction * self.radian2angle
        if l1 == l2:
            return x + l1 * math.cos(direct_rad), y + l1 * math.sin(direct_rad), direction
        elif l1 > l2:
            alpha = direction * self.radian2angle
            theta = abs(l1 - l2) / (2 * r)
            direction -= theta / self.radian2angle
            if direction < -180:
                direction += 360
            tmp = (2 * r * min(l1, l2)) / abs(l1 - l2)
            p_x = x + (r + tmp) * math.cos(math.pi / 2 - alpha)
            p_y = y - (r + tmp) * math.sin(math.pi / 2 - alpha)
            new_x = p_x + (r + tmp) * math.cos(math.pi / 2 + alpha - theta)
            new_y = p_y + (r + tmp) * math.sin(math.pi / 2 + alpha - theta)
            return new_x, new_y, direction
        elif l1 < l2:
            alpha = direction * self.radian2angle
            theta = l2 / (2 * r)
            direction += theta / self.radian2angle
            if direction > 180:
                direction -= 360
            tmp = (2 * r * min(l1, l2)) / abs(l1 - l2)
            p_x = x - (r + tmp) * math.cos(alpha - math.pi / 2)
            p_y = y - (r + tmp) * math.sin(alpha - math.pi / 2)
            new_x = p_x - (r + tmp) * math.cos(3 * math.pi / 2 - alpha - theta)
            new_y = p_y + (r + tmp) * math.sin(3 * math.pi / 2 - alpha - theta)
            return new_x, new_y, direction

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        time.sleep(1 / self.fps)

        chaser = self.agents[AgentType.Chaser]
        runner = self.agents[AgentType.Runner]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)

            # chaser circle object
            chaser_obj = rendering.make_circle(chaser.r, 30, True)
            chaser_obj.set_color(*chaser.color)
            self.viewer.add_geom(chaser_obj)
            self.chaser_trans = rendering.Transform()
            chaser_obj.add_attr(self.chaser_trans)

            # chaser wheel object
            left_wheel = rendering.make_polygon([
                (chaser.wheel_radius, chaser.r),
                (chaser.wheel_radius, chaser.r + chaser.wheel_width),
                (-chaser.wheel_radius, chaser.r + chaser.wheel_width),
                (-chaser.wheel_radius, chaser.r)
            ])
            right_wheel = rendering.make_polygon([
                (chaser.wheel_radius, -chaser.r - chaser.wheel_width),
                (chaser.wheel_radius, -chaser.r),
                (-chaser.wheel_radius, -chaser.r),
                (-chaser.wheel_radius, -chaser.r - chaser.wheel_width)
            ])
            left_wheel.set_color(*chaser.left_wheel_color)
            right_wheel.set_color(*chaser.right_wheel_color)
            self.left_trans = rendering.Transform()
            left_wheel.add_attr(self.left_trans)
            self.right_trans = rendering.Transform()
            right_wheel.add_attr(self.right_trans)
            self.viewer.add_geom(left_wheel)
            self.viewer.add_geom(right_wheel)

            # direction arrow object
            direct_obj = rendering.FilledPolygon([
                (0.5 * chaser.r, 0.15 * chaser.r),
                (chaser.r, 0),
                (0.5 * chaser.r, -0.15 * chaser.r)
            ])
            self.line_trans = rendering.Transform()
            direct_obj.set_color(1, 1, 1)
            direct_obj.add_attr(self.line_trans)
            self.viewer.add_geom(direct_obj)

            # runner circle object
            runner_obj = rendering.make_circle(runner.r, 30, True)
            runner_obj.set_color(*runner.color)
            self.viewer.add_geom(runner_obj)
            self.runner_trans = rendering.Transform()
            runner_obj.add_attr(self.runner_trans)

        self.chaser_trans.set_translation(chaser.x, chaser.y)
        self.runner_trans.set_translation(runner.x, runner.y)
        self.line_trans.set_translation(chaser.x, chaser.y)
        self.line_trans.set_rotation(chaser.direction * self.radian2angle)
        self.left_trans.set_translation(chaser.x, chaser.y)
        self.left_trans.set_rotation(chaser.direction * self.radian2angle)
        self.right_trans.set_translation(chaser.x, chaser.y)
        self.right_trans.set_rotation(chaser.direction * self.radian2angle)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = PuckWorldWheel(fps=60)
    chaser = AgentWithWheel(50, 50, 30, (1, 0, 0), AgentType.Chaser)
    runner = AgentWithWheel(300, 300, 30, (0, 0, 1), AgentType.Runner)

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
