from gym.envs.registration import register


register(
    id='puck-world-v0',
    entry_point='puck_world.envs:PuckWorld'
)

register(
    id='puck-world-wheel-v0',
    entry_point='puck_world.envs:PuckWorldWheel'
)