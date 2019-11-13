"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Continuous version by Ian Danforth:

https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8

Edited as there was some difference between dynamics and eqns of motion from:
https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf

Also added damping

"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import pyglet


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    MAX_POS = 2.5
    MAX_ANGLE = np.pi
    MAX_VEL_1 = 1.0
    MAX_VEL_2 = 4 * np.pi
    swingup = False

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.05  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.angular_damping = 0.00
        self.linear_damping = 0.00

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = self.MAX_POS

        # theta offset
        self.theta_offset = np.pi

        # Observation mode
        self.symbolic = True

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        self.high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        #self.high = np.array([self.MAX_POS, self.MAX_VEL_1, self.MAX_ANGLE,  self.MAX_VEL_2])
        self.observation_space = spaces.Box(-self.high, self.high)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        #self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        #self.swingup = False
        self.steps_beyond_done = None

    def set_params(self, pole_mass, pole_length, cart_mass, damping=0.0):
        self.masscart = cart_mass
        self.masspole = pole_mass
        self.angular_damping = damping
        self.linear_damping = 0.0
        self.total_mass = (self.masspole + self.masscart)
        self.length = pole_length * 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        theta -= self.theta_offset - np.pi
        q_dot = np.array([[x_dot], [theta_dot]])

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        mc = self.masscart
        mp = self.masspole
        l = self.length
        g = self.gravity
        b1 = self.linear_damping
        b2 = self.angular_damping

        # Inertia matrix
        tmp = l * (mc + mp * sintheta * sintheta)

        thetaacc = (-force * costheta -
                    mp * l * theta_dot * theta_dot * sintheta * costheta -
                    (mc + mp) * g * sintheta + b1 * x_dot * costheta - (mc + mp) * b2 * theta_dot / (mp * l)) / tmp

        xacc = (force * l + mp * l * sintheta * (l * theta_dot * theta_dot + g * costheta) +
                costheta * b2 * theta_dot - l * b1 * x_dot) / tmp

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta += self.theta_offset - np.pi
        return np.asarray([x, x_dot, theta, theta_dot])

    def step(self, action):
        #assert self.action_space.contains(action), \
        #    "%r (%s) invalid" % (action, type(action))
        action = np.clip(action, -self.action_space.high[0], self.action_space.high[0])
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)

        #self.state = self.stepPhysics_old(force)
        self.state = self.stepPhysics(force)
        self.state[2] = wrap(self.state[2], -np.pi, np.pi)
        #self.state = np.clip(self.state, -self.high, self.high)

        x, x_dot, theta, theta_dot = self.state

        #done = False
        #theta_d = theta / np.pi
        #cost = 0.1 * x **2 + theta_d**2 + .01 * theta_dot**2 + .1 * x_dot

        #reward = -cost

        theta -= self.theta_offset
        theta = wrap(theta, -np.pi, np.pi)
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        if self.swingup:
            done = False

        observation = np.array(self.state)
        return observation, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        #self.state[2] = self.np_random.uniform(low=-0.01*np.pi, high=0.01*np.pi, size=(1,))
        self.state[2] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))

        if self.swingup:
            self.state[2] += np.pi

        self.state[2] += self.theta_offset
        #self.state[2] += np.pi
        self.state[2] = wrap(self.state[2], -np.pi, np.pi)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        x[2] -= self.theta_offset
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        x[2] += self.theta_offset
        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()


class ContinuousCartPoleSwingupEnv(ContinuousCartPoleEnv):
    swingup = True


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


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)