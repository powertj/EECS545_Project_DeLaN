"""Double pendulum modified from gym acrobot task:  https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py
"""
import numpy as np
from numpy import sin, cos, pi

from gym import core, spaces
from gym.utils import seeding


class DoublePendulumEnv(core.Env):

    """
    Manipulator equations from

    https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    dt = .05

    LINK_LENGTH_1 = 0.5 # [m]
    LINK_LENGTH_2 = 0.5  # [m]
    LINK_MASS_1 = 0.5 #: [kg] mass of link 1
    LINK_MASS_2 = 0.5 #: [kg] mass of link 2
    LINK_COM_POS_1 = LINK_MASS_1 / 2  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = LINK_MASS_2 / 2  #: [m] position of the center of mass of link 2
    LINK_MOI = 3.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 4 * np.pi

    MAX_TORQUE = 1.0
    torque_noise_max = 0.
    swingup = False

    def __init__(self):
        self.viewer = None
        self.state = None
        self.seed()

        self.action_dim = 1
        self.g = 9.8
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2])
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-self.MAX_TORQUE,
                                       high=self.MAX_TORQUE,
                                       shape=(self.action_dim,),
                                       dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        high = np.array([np.pi / 2.0, np.pi/ 2.0, 0.5, .5])
        self.state = self.np_random.uniform(low=-high, high=high)
        if self.swingup:
            self.state[0] += np.pi
        return self._get_ob()

    def step(self, a):
        s = self.state
        a = np.expand_dims(np.asarray(a), axis=1)
        # Add noise to the force action
        # if self.torque_noise_max > 0:
        #    torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Perform step
        s[0] += np.pi
        ns = s + self.dt * self._dsdt(s, a)
        ns[0] -= np.pi
        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)

        # Bound to max velocity -- can get rid of this maybe?
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = False
        reward = None

        return (self._get_ob(), reward, terminal, {})

    def _get_ob(self):
        s = self.state
        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def _terminal(self):
        s = self.state
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.)

    def _dsdt(self, s, u):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = self.g

        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]

        # Trigonometric identities
        c2 = np.cos(theta2)
        s1 = np.sin(theta1)
        s2 = np.sin(theta2)
        s12 = np.sin(theta1 + theta2)

        # Calculate matrices for manipulator equations
        H = np.array([
            [I1 + I2 + m2 * np.square(l1) + 2 * m2 * l1 * lc2 * c2, I2 + m2 * l1 * lc2 * c2],
            [I2 + m2 * l1 * lc2 * c2, I2]
        ])

        C = np.array([
            [-2 * m2 * l1 * lc2 * s2 * dtheta2, -m2 * l1 * lc2 * s2 * dtheta2],
            [m2 * l1 * lc2 * s2 * dtheta1, 0.0]
        ])

        G = np.array([
            [(m1 * lc1 + m2 * l1) * g * s1 + m2 * g * l2 * s12],
            [m2 * g * l2 * s12]
        ])

        if self.action_dim == 1:
            B = np.array([
                [1.],
                [0.]
            ])
        else:
            B = np.eye(2)

        dq = np.array([
            [dtheta1],
            [dtheta2],
        ])

        # Solve manipulator equations for angular accelerations
        lhs = B @ u - (C @ dq + G)
        ddq = np.linalg.solve(H, lhs)

        ddtheta1 = ddq[0, 0]
        ddtheta2 = ddq[1, 0]

        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state
        #s[0] += np.pi

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None:
            return None

        p1 = [-self.LINK_LENGTH_1 *
              np.cos(s[0] + np.pi), self.LINK_LENGTH_1 * np.sin(s[0] + np.pi)]

        p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + np.pi + s[1]),
              p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + np.pi + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0] + np.pi/2, s[0] + s[1] + np.pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(8., .3, .3)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(0, 0, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)