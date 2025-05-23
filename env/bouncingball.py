import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class BouncingBall(gym.Env):
    def __init__(self, config=None):
        self.v = 0  # velocity
        self.c = 0  # cost/hit counter
        self.p = 0  # position
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.dt = 0.1
        self.seed()
        self.mode=1
        if config is not None:
            self.dt = config.get("tau", 0.1)

    def reset(self):
        if self.mode==0:
            self.p = 7 + self.np_random.uniform(5, 9)
            self.v = self.np_random.uniform(-1,1)
        elif self.mode==1:
            self.p=self.np_random.uniform(5,9)
            self.v=self.np_random.uniform(-1, 1)
        elif self.mode==2:
            self.p=self.np_random.uniform(5.05,5.1)
            self.v=self.np_random.uniform(-0.05, 0.05)
        elif self.mode==3:
            self.p=self.np_random.uniform(1,15)
            self.v=self.np_random.uniform(-8, 8)
        else:
            self.p=5.1
            self.v=1
        return np.array((self.p, self.v))

    def step(self, action):
        done = False

        cost = 0
        # v_prime = self.v - 9.81 * self.dt
        # p_prime = max(self.p + self.dt * v_prime, 0)
        v_prime = self.v
        p_prime = self.p
        if v_prime <= 0 and p_prime <= 0:
            v_prime = -(0.90) * v_prime
            p_prime = 0
            if v_prime <= 1:
                done = True
                # cost += -1000
        if v_prime <= 0 and p_prime > 4 and action == 1:
            v_prime = v_prime - 4
            p_prime = 4
        if v_prime > 0 and p_prime > 4 and action == 1:
            v_prime = -(0.9) * v_prime - 4
            p_prime = 4
        # v_second = v_prime - 9.81 * dt
        # p_second = p_prime + dt * v_prime
        self.p = p_prime
        self.v = v_prime
        cost += -1 if action == 1 else 0
        if not done:
            cost += 1
        return np.array((self.p, self.v)), cost, done, {}

    @staticmethod
    def calculate_successor(state, action):
        p, v = state
        dt = 0.1
        done = False

        cost = 0
        v_prime = v - 9.81 * dt
        p_prime = max(p + dt * v_prime, 0)
        if v_prime <= 0 and p_prime <= 0:
            v_prime = -(0.90) * v_prime
            p_prime = 0
            if v_prime <= 1:
                done = True
                # cost += -1000
        if v_prime <= 0 and p_prime > 4 and action == 1:
            v_prime = v_prime - 4
            p_prime = 4
        if v_prime > 0 and p_prime > 4 and action == 1:
            v_prime = -(0.9) * v_prime - 4
            p_prime = 4
        # v_second = v_prime - 9.81 * dt
        # p_second = p_prime + dt * v_prime
        cost += -1 if action == 1 else 0
        if not done:
            cost += 1
        if v_prime < 7 and v_prime > -7 and p < 1:
            done = True
        return np.array((p_prime, v_prime)), cost, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed) #get a random generator and a random seed
        return [seed]


if __name__ == '__main__':
    env = BouncingBall()
    state = env.reset()
    position_list = [state[0]]
    print(state)
    done = False
    i = 0
    while True:
        state, cost, done, _ = env.step(0)
        position_list.append(state[0])
        print(state)
        i += 1
        if i > 500:
            break
        if done:
            print("done")
            break
    import plotly.graph_objects as go

    fig = go.Figure()
    trace1 = go.Scatter(x=list(range(len(position_list))), y=position_list, mode='markers', )
    fig.add_trace(trace1)
    fig.write_html("bouncingballboucingballtrace.html")