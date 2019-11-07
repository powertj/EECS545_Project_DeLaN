""" Reacher modified from gym acrobot task:  https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py
"""
import numpy as np
from gym import core, spaces
from gym_cenvs.envs.double_pendulum import DoublePendulumEnv


class ReacherEnv(DoublePendulumEnv):

    """
    Reacher is essentially the same as double pendulum but with both joints actuated, no gravity, and a variable goal

    """
    def __init__(self):
        super(ReacherEnv, self).__init__()
        self.action_dim = 2
        self.g = 0.0
        self.action_space = spaces.Box(low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(2,),  dtype=np.float32)

    def reset(self):
        high = np.array([np.pi, np.pi, 0., 0.])
        self.state = self.np_random.uniform(low=-high, high=high)
        return self._get_ob()
