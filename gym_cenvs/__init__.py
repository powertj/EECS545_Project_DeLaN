from gym.envs.registration import register

register(
    id='DoublePendulum-v0',
    entry_point='gym_cenvs.envs:DoublePendulumEnv',
)
register(
    id='Reacher-v0',
    entry_point='gym_cenvs.envs:ReacherEnv',
)
