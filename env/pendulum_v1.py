from os import path

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

current_folder = path.abspath(path.dirname(__file__))


class Pendulum(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        g = 10.0
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.discrete_actions = True
        self.state=np.array([])
        self.mode=2
        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        if not self.discrete_actions:
            self.action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque, shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        if not self.discrete_actions:
            u = np.clip(action, -self.max_torque, self.max_torque)
        else:
            u = 0
            if action == 1:
                u = -self.max_torque
            if action == 2:
                u = self.max_torque
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = angle_normalize(newth)

        self.state = np.array([newth, newthdot])
        return self.state, -costs, False, {}

    def reset(self):
        #train mode
        if(self.mode==0):
            high = np.array([np.pi, 4])
            self.state = self.np_random.uniform(low=-high, high=high)
        #high = np.array([0.05, 0.05])
        #high=np.array([0.0055,0.0])
        #low=np.array([0,-0.01])
        #self.state = self.np_random.uniform(low=low, high=high)
        #self.state = self.np_random.uniform(low=-high, high=high)
        #self.last_u = None
        #verify mode
        elif(self.mode==1):
            # self.state=np.array([-0.05,0.1])
            self.state = np.array([1e-5, 0.0002])
        else:
            high = np.array([0.05, 0.05])
            self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta, thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), f"{current_folder}/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)  # normalise the angle between \pi and -\pi





if __name__ == '__main__':
    env = PendulumEnv()
    env.reset()

    done = False
    while not done:
        obs, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        # time.sleep(1 / 30)