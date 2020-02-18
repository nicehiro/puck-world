class Agent:
    def __init__(self, init_x, init_y, radius, color, agent_type):
        self.x = init_x
        self.y = init_y
        self.r = radius
        self.color = color
        self.type = agent_type

    def act(self, state):
        pass


class AgentWithWheel(Agent):
    def __init__(self, init_x, init_y, radius, color, agent_type):
        super().__init__(init_x, init_y, radius, color, agent_type)
        self.wheel_radius = 0.4 * radius
        self.wheel_width = 0.1 * radius
        self.left_wheel_color = (0, 0, 0)
        self.right_wheel_color = (0, 0, 0)
        # unit: px/s
        self.left_v = 40
        self.right_v = 20
        # direction: 0~360
        self.direction = 0
        self.v_unit = 50

    def set_left_v(self, left_v):
        self.left_v = left_v * self.v_unit

    def set_right_v(self, right_v):
        self.right_v = right_v * self.v_unit

    def act(self, state):
        pass
