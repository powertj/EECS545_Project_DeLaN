from gym.envs.registration import register

register(
    id='DoublePendulum-v0',
    entry_point='gym_cenvs.envs:DoublePendulumEnv',
)
register(
    id='Reacher-v0',
    entry_point='gym_cenvs.envs:ReacherEnv',
)
register(
    id='ContinuousCartpole-v0',
    entry_point='gym_cenvs.envs:ContinuousCartPoleEnv',
)
register(
    id='ContinuousCartpole-v1',
    entry_point='gym_cenvs.envs:ContinuousCartPoleSwingupEnv',
)