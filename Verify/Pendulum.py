import string

import pandas as pd
from matplotlib import patches
from matplotlib.collections import PatchCollection

from Verify.AbstractTraining.ppo_discrete_main import *
import math
import time
from Tools.abstractor import *
from Tools.Intersector import *
from Tools.Graph import *
from queue import Queue
import numpy as np
from numpy import matlib
import cupy as cp
import cupyx as cpx
from py4j.java_collections import ListConverter
from py4j.java_gateway import JavaGateway
import random
import stormpy
import os
from scipy.optimize import minimize_scalar
import matplotlib.colors as mcolors
class Transition_sy:
    def __init__(self, args, divide_tool):
        self.agent = PPO_discrete(args)
        load(self.agent,args)
        self.action = [0, 1, 2]
        # self.initial_state = [-0.05, 0.1]  # [0.5, 0.001, 0.1, 0.001]   [0.001, 0.001, 0, 0.001]
        self.initial_state = [1e-5, 0.0002]  # [0.5, 0.001, 0.1, 0.001]   [0.001, 0.001, 0, 0.001]
        self.abstract_initial_states = []
        self.initial_state_region = [-0.54999,3.60001,-0.450001,3.849999]
        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 300
        self.divide_tool = divide_tool

        self.rtree_size = 0
        self.graph = Graph()
        # dynamics 参数
        self.max_speed = 10
        self.max_torque = 2.
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.
        self.l = 1.

        self.round_rate = 100000000000000
        self.label = {}
        self.unsafe = []

        self.k_steps = [10, 20, 35, 50, 70, 90, 120, 150]
        self.sim_prob = []
        self.veri_prob = []
        self.timecost=[]
        self.map={}
    def list_to_str(self,state_list):
        obj_str = ','.join([str(_) for _ in state_list])
        return obj_str
    def angle_normalize(self, x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)  # normalise the angle between \pi and -\pi
    # 计算下一个状态可能处于的最大bound
    def next_abstract_domain(self, abstract_stat, action):  # 获取下一个状态的数据范围
        theta0, theta_dot0, theta1, theta_dot1 = abstract_stat
        dt=self.dt
        # get the value of torque from action
        u = 0
        if action == 1:
            u = -self.max_torque
        if action == 2:
            u = self.max_torque
        #scipy求解
        # def thdot_compute_min(th):
        #     return (-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3. / (self.m * self.l ** 2) * u) * self.dt
        # def thdot_compute_max(th):
        #     return -((-3 * self.g / (2 * self.l) * np.sin(th + np.pi) + 3. / (self.m * self.l ** 2) * u) * self.dt)
        # new_theta_dot0=minimize_scalar(thdot_compute_min, bounds=(theta0,theta1),method='bounded').fun+theta_dot0
        # new_theta_dot1=-(minimize_scalar(thdot_compute_max, bounds=(theta0,theta1),method='bounded').fun)+theta_dot1
        #
        # new_theta0=theta0+new_theta_dot0*self.dt
        # new_theta1=theta1+new_theta_dot1*self.dt
        # new_theta_dot0=np.clip(new_theta_dot0, -self.max_speed, self.max_speed)
        # new_theta_dot1=np.clip(new_theta_dot1, -self.max_speed, self.max_speed)
        # new_theta0 = self.angle_normalize(new_theta0)
        # new_theta1 = self.angle_normalize(new_theta1)

        #近似写法
        # new_theta_dot0 = theta_dot0 + dt * (-3 * self.g * math.sin(np.pi + theta0) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
        # new_theta_dot1 = theta_dot1 + dt * (-3 * self.g * math.sin(np.pi + theta1) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
        # new_theta0 = theta0 + new_theta_dot0 * dt
        # new_theta1 = theta1 + new_theta_dot1 * dt

        # #精确的单调性分析
        # #关于角度单减
        if theta1<-np.pi/2 or theta0 > np.pi / 2:
            new_theta_dot0 = theta_dot0 + dt * (3 * self.g * math.sin(theta1) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
            new_theta_dot1 = theta_dot1 + dt * (3 * self.g * math.sin(theta0) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
        #在-pi/2处取得最小值，最大值在两边界取
        elif theta0<-np.pi/2 and theta1>-np.pi/2:
            new_theta_dot0 = theta_dot0 + dt * (-3 * self.g / 2 * self.l + 3 * u / (self.m * self.l ** 2))
            new_theta_dot1 = theta_dot1 + dt * (3 * self.g * max(math.sin(theta0),math.sin(theta1)) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
        #单调递增
        elif theta0>-np.pi/2 and theta1<np.pi/2:
            new_theta_dot0 = theta_dot0 + dt * (3 * self.g * math.sin(theta0) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
            new_theta_dot1 = theta_dot1 + dt * (3 * self.g * math.sin(theta1) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
        #在pi/2处取得最大值，最小值在两边取
        elif theta0<np.pi/2 and theta1>np.pi/2:
            new_theta_dot0 = theta_dot0 + dt * (3 * self.g * min(math.sin(theta0), math.sin(theta1)) / 2 * self.l + 3 * u / (self.m * self.l ** 2))
            new_theta_dot1 = theta_dot1 + dt * (3 * self.g  / 2 * self.l + 3 * u / (self.m * self.l ** 2))

        new_theta0 = theta0 + new_theta_dot0 * dt
        new_theta1 = theta1 + new_theta_dot1 * dt

        # 限制越界的情况
        new_theta_dot0 = np.clip(new_theta_dot0, -self.max_speed, self.max_speed)
        new_theta_dot1 = np.clip(new_theta_dot1, -self.max_speed, self.max_speed)
        new_theta0 = self.angle_normalize(new_theta0)
        new_theta1 = self.angle_normalize(new_theta1)

        new_state = [new_theta0,  new_theta_dot0, new_theta1,new_theta_dot1]
        return new_state

    def get_next_states(self, decision_part, transition_part):
        # the return:successors with correspond prob
        suc_with_prob = []
        # compute the prob of taking each action over the states

        curS = torch.unsqueeze(torch.tensor(decision_part, dtype=torch.float), 0)
        prob = self.agent.actor(curS).detach().numpy().flatten()

        # compute the successors under each action
        for i in range(len(self.action)):
            suc_with_prob.append([self.next_abstract_domain(transition_part, self.action[i]), prob[i]])
        # suc_with_prob.append([self.next_abstract_domain(transition_part, self.action[1]), 1])
        return suc_with_prob
    #
    # def refine_branch(self, current):
    #     # return: branchs in MDP,each branch has a decision part and an area to compute the successors
    #     braches = []
    #     dim = int(len(current)/2)
    #     Decision_area = self.divide_tool.intersection(current)
    #     for part in Decision_area:
    #         ith_part = str_to_list(part)
    #         intersection = get_intersection(dim, current, ith_part)
    #         flag=True
    #         for j in range(dim):
    #             if(intersection[j].upper==intersection[j].lower):
    #                 flag=False
    #         if(flag==False):
    #             continue
    #         transition_area = [None] * dim*2
    #         for j in range(dim):
    #             transition_area[j] = intersection[j].lower
    #             transition_area[j + dim] = intersection[j].upper
    #         braches.append([ith_part, transition_area])
    #     return braches
    def refine_branch(self, current):
        # return: branchs in MDP,each branch has a decision part and an area to compute the successors
        braches = []
        dim = int(len(current) / 2)
        Decision_area = self.divide_tool.intersection(current)
        for part in Decision_area:
            ith_part = str_to_list(part)
            intersection = get_intersection(dim, current, ith_part)
            flag = True
            for j in range(dim):
                if (intersection[j].upper == intersection[j].lower):
                    flag = False
            if (flag == False):
                continue
            braches.append([ith_part, ith_part])
        return braches
    def labeling(self, abstract_state):
        state_list = str_to_list(abstract_state)
        if state_list[0] >= -np.pi / 6 - 0.01 and state_list[2] <= np.pi / 6 + 0.01:
            return "safe"
        else:
            return "unsafe"

    def CreateMDP(self):
        abstract_state_count = 0
        bfs = Queue()
        builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False,
                                              has_custom_row_grouping=True, row_groups=0)
        # record the number of rows in the transition matrix
        row = 0
        self.abstract_initial_states = self.divide_tool.intersection(self.initial_state_region)
        # self.abstract_initial_states.append(self.divide_tool.get_abstract_state(self.initial_state))
        # self.abstract_initial_states.append(self.list_to_str(self.initial_state_region))
        mapping = {}
        state_ranges = []
        # 初始状态添加标签
        t0 = time.time()
        for abstract_state in self.abstract_initial_states:
            if not mapping.__contains__(abstract_state):
                mapping[abstract_state] = abstract_state_count
                self.label[mapping[abstract_state]] = self.labeling(abstract_state)
                bfs.put(str_to_list(abstract_state))
                if (self.labeling(abstract_state) == "safe"):
                    state_ranges.append(str_to_list(abstract_state))
                else:
                    self.unsafe.append(abstract_state_count)
                abstract_state_count += 1
            else:
                print("The same initial state has exsisted:", mapping[abstract_state])
        print("Steps :", 0, "   |   ", "Transition Size :", abstract_state_count, "   |   ", "Time Cost:",0)
        tmp = Queue()
        for i in range(self.limited_depth):
            if bfs.empty():
                break
            while not bfs.empty():
                # 取出同一层的所有节点
                state_node = bfs.get()
                tmp.put(state_node)
            # 计算该层所有节点的后继节点
            while not tmp.empty():
                current = tmp.get()
                if self.label[mapping[self.list_to_str(current)]]== "unsafe":
                    builder.new_row_group(row)
                    builder.add_next_value(row, mapping[self.list_to_str(current)], 1)
                    row+=1
                    continue
                current_branches = self.refine_branch(current)
                builder.new_row_group(row)  # current states as new row
                for j in range(len(current_branches)):
                    successor = self.get_next_states(current_branches[j][0], current_branches[j][1])
                    for k in range(len(successor)):
                        # if the successor is not visited and safe, give it a state index, and add it to the bfs queue
                        successor_str=self.list_to_str(successor[k][0])
                        if not mapping.__contains__(successor_str):
                            mapping[successor_str] = abstract_state_count
                            self.label[mapping[successor_str]] = self.labeling(successor_str)
                            if self.label[mapping[successor_str]] == 'unsafe':
                                self.unsafe.append(mapping[successor_str])
                            abstract_state_count += 1
                            bfs.put(successor[k][0])
                            state_ranges.append(successor[k][0])
                        #添加该分支下的后续迁移，但是注意不保证在该分支下，两个动作迁移至同一个状态，可能有bug
                        builder.add_next_value(row, mapping[successor_str], successor[k][1])
                    row+=1
            t3 = time.time()
            tt = t3 - t0
            print("Steps :", i + 1, "   |   ", "Transition Size :", abstract_state_count, "   |   ", "Time Cost:", tt)
            self.timecost.append(tt)
        while not bfs.empty():
            state_node = bfs.get()
            builder.new_row_group(row)
            builder.add_next_value(row, mapping[self.list_to_str(state_node)], 1)
            row += 1
        if(len(self.unsafe)==0):
            builder.new_row_group(row)
            builder.add_next_value(row, abstract_state_count, 1)
            self.unsafe.append(abstract_state_count)
            abstract_state_count+=1
            row += 1
        transition_matrix = builder.build()
        state_labeling = stormpy.storage.StateLabeling(abstract_state_count)
        labels = {'init', 'unsafe'}
        for label in labels:
            state_labeling.add_label(label)
        state_labeling.set_states('unsafe', stormpy.BitVector(abstract_state_count, self.unsafe))
        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
        mdp = stormpy.storage.SparseMdp(components)
        print(mdp)
        self.map=mapping
        # trajectories=self.generate_trajectories(max_steps=self.limited_depth+1)
        # self.plot_state_ranges(state_ranges,trajectories)
        return mdp

    def ModelCheckingbyStorm(self, mdp, step):
        print("-------------Verifying the  MDP in {} steps----------".format(step))
        t1 = time.time()
        formula_str = "Pmax=? [(true U<=" + str(step) + "\"unsafe\") ]"
        # formula_str = "P=? [(true U \"unsafe\") ]"
        properties = stormpy.parse_properties(formula_str)
        result = stormpy.model_checking(mdp, properties[0])
        self.veri_prob.append(1 - result.get_values()[0])
        # print("The result of Check MDP by Storm :", 1 - result.get_values()[0])
        t2 = time.time()
        print("Check the DTMC by Storm, cost time:", "[", t2 - t1, "]")
        return result.get_values()

    def plot_state_ranges(self,state_ranges,trajectories):
        fig, ax = plt.subplots(figsize=(64, 48))
        # 使用PatchCollection批量绘制矩形
        patches_list = []
        for state_range in state_ranges:
            x_min,  y_min, x_max,y_max = state_range
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                     facecolor='none', linestyle='dashed')
            patches_list.append(rect)

        patch_collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(patch_collection)
        #绘制轨迹
        for trajectory in trajectories:
            trajectory = np.array(trajectory)
            x_vals, y_vals = trajectory[:, 0], trajectory[:, 1]
            #ax.scatter(x_vals, y_vals, alpha=0.5)  # 设置轨迹透明度为0.5
            ax.plot(x_vals, y_vals, alpha=0.8)  # 设置轨迹透明度为0.5

        initial_state_low = np.array([-0.05, -0.05])
        initial_state_high = np.array([0.05, 0.05])
        plt.plot([initial_state_low[0], initial_state_high[0], initial_state_high[0], initial_state_low[0],
                  initial_state_low[0]],
                 [initial_state_low[1], initial_state_low[1], initial_state_high[1], initial_state_high[1],
                  initial_state_low[1]],
                 color='blue', linestyle='--', label='Initial State Area')

        #safe 区域
        safe_low = np.array([-np.pi/6, -8])
        safe_high = np.array([np.pi/6, 8])
        plt.plot([safe_low[0], safe_high[0], safe_high[0], safe_low[0],
                  safe_low[0]],
                 [safe_low[1], safe_low[1], safe_high[1], safe_high[1],
                  safe_low[1]],
                 color='green', linestyle='--', label='safe Area')
        # 设置图像显示的范围
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-8, 8)

        # 设置坐标轴标签
        ax.set_xlabel('theta')
        ax.set_ylabel('thetadot')

        # 显示图像
        plt.grid(True)
        plt.title('State Ranges')
        plt.savefig('./pd.png')  # 保存图片

    def generate_trajectories(self, num_episodes=50, max_steps=40):
        env = gym.make(args.envName)
        trajectories = []
        for _ in range(num_episodes):
            state = env.reset()
            s=state
            trajectory = [state]
            for _ in range(max_steps):
                s = clip(s, [[-np.pi+0.00001, -7.9], [np.pi-0.00001, 7.9]])
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                next_state, reward, done, _ = env.step(a)
                trajectory.append(next_state)
                s = next_state
                if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01 or done:
                    break
            trajectories.append(trajectory)
        return trajectories
    def teststep(self, statespace_1):
        divide_tool = self.divide_tool
        env = gym.make(args.envName)
        agent = PPO_discrete(args)
        steps = []
        # Xmin=0
        # Xmax=0
        # Vmin=0
        # Vmax=0
        for i in range(2):
            s = env.reset()
            abs = divide_tool.get_abstract_state(s)
            abs = str_to_list(abs)
            step = 0
            while True:
                print("step:---", step, env.delta_v, env.delta_x)
                print("abs:", abs)
                # Xmin=min(Xmin,s[1])
                # Xmax=max(Xmax,s[1])
                # Vmin = min(Vmin,s[0])
                # Vmax = max(Vmax,s[0])
                s = clip(s, statespace_1)
                abstract_s = divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = agent.choose_action(abstract_s)
                # print(a)
                s_, r, done, info = env.step(a)
                s = s_
                p_current = [abs[0], abs[2], abs[1], abs[3]]
                p_next_domain = self.next_abstract_domain(p_current, a)
                abs_ = [p_next_domain[0], p_next_domain[2], p_next_domain[1], p_next_domain[3]]
                abs = abs_
                step += 1
                if s[1] <= 0:
                    steps.append(step)
                    break
        step_mean = torch.mean(torch.Tensor(steps))
        print("step_mean=", step_mean)
        # print(Xmin,Xmax,Vmax,Vmin)
    def Simulate(self, step,area, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env=Pendulum()
        np_random, seed = seeding.np_random()
        # env.mode=2
        unsafecount = 0
        for i in range(10000):
            low = np.array([area[0], area[1]])
            high = np.array([area[2], area[3]])
            state = np_random.uniform(low=low, high=high)
            s=state
            env.state=state
            for k in range(step):
                if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01:
                    unsafecount += 1
                    break
                s = clip(s, statespace_1)
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                s_, r, done, info = env.step(a)
                s = s_
                if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01:
                    unsafecount += 1
                    break
        print("k=", step,"The Simulated Unsafe Prob:",  unsafecount / 10000)
        return unsafecount / 10000

    def Simulate_lowerbound(self, step,area, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env=Pendulum()
        max_prob=0.0
        np_random, seed = seeding.np_random()
        for i in range(10):
            low = np.array([area[0], area[1]])
            high = np.array([area[2],area[3]])
            state = np_random.uniform(low=low, high=high)
            unsafecount = 0
            for j in range(500):
                s=state
                env.state=state
                for k in range(step):
                    s = clip(s, statespace_1)
                    if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01:
                        unsafecount += 1
                        break
                    abstract_s = self.divide_tool.get_abstract_state(s)
                    abstract_s = str_to_list(abstract_s)
                    abstract_s = np.array(abstract_s)
                    a, logprob = self.agent.choose_action(abstract_s)
                    s_, r, done, info = env.step(a)
                    s = s_
                    if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01:
                        unsafecount += 1
                        break
            max_prob=max(max_prob,unsafecount/500)
        print("k=", step, "The Simulated Max Probability:", max_prob)
        return max_prob
    def plot_prob_area(self,area,res_prob,file_path):
        def hex_to_rgb(hex_color):
            """Convert hex color to RGB."""
            hex_color = hex_color.lstrip('#')
            return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]

        def rgb_to_hex(rgb):
            """Convert RGB color to hex."""
            return '#{:02x}{:02x}{:02x}'.format(*rgb)

        def generate_gradient(start_hex, end_hex, num_colors):
            """Generate a gradient of colors between two hex colors."""
            start_rgb = np.array(hex_to_rgb(start_hex))
            end_rgb = np.array(hex_to_rgb(end_hex))

            gradient = np.linspace(start_rgb, end_rgb, num_colors)
            gradient_hex = [rgb_to_hex(tuple(map(int, color))) for color in gradient]

            return gradient_hex

        start_color = '#8E7FB8'
        end_color = '#A2C9AE'

        # Generate gradient colors
        num_colors = 6  # Number of colors in the gradient
        gradient_colors = generate_gradient(start_color, end_color, num_colors)
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', gradient_colors)
        norm = plt.Normalize(vmin=0.0, vmax=1)

        probabilities = [res_prob[self.map[rect_str]] for rect_str in self.abstract_initial_states]
        avg_prob = np.mean(probabilities)
        min_prob = np.max(probabilities)
        min_prob_index = probabilities.index(min_prob)

        fig, ax = plt.subplots()

        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=res_prob[self.map[rect_str]]
            x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
            width = x_max - x_min
            height = y_max - y_min
            # 使用概率值确定矩形的颜色
            color_idx = int(norm(prob) * (num_colors - 1))
            # color = gradient_colors[color_idx]
            color = plt.cm.rainbow(prob)
            # if(self.map[rect_str] == min_prob_index):
            #     min_rect_patch = patches.Rectangle((x_min, y_min), width, height,linewidth=1, edgecolor='red', facecolor=color)
            #     continue
            # color = plt.cm.viridis(prob)  # viridis 是一个颜色映射表，可以根据需要更改
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
        # for rect_str in self.abstract_initial_states:
        #     if (self.map[rect_str] == min_prob_index):
        #         prob = res_prob[self.map[rect_str]]
        #         x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
        #         width = x_max - x_min
        #         height = y_max - y_min
        #         # 使用概率值确定矩形的颜色
        #         color_idx = int(norm(prob) * (num_colors - 1))
        #         color = gradient_colors[color_idx]
        #         min_rect_patch = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='red',facecolor=color)
        #         ax.add_patch(min_rect_patch)
        #         break
        ax.set_xlim(area[0], area[2])
        ax.set_ylim(area[1], area[3])
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        plt.grid(True)
        plt.savefig(file_path)  # 保存图片
        print(f"The heatmap of Verification  have been save to{file_path}")
    def plot_correctness(self,region,file_path):
        def Simulate_prob( step,state, statespace_1):
            # print("---------Start Simulating in {} steps---------".format(step))
            # env = gym.make(args.envName)
            env = Pendulum()
            # area = map(float, state.split(','))
            area = str_to_list(state)
            max_prob = 0.0
            np_random, seed = seeding.np_random()
            unsafecount = 0
            for i in range(100):
                low = np.array([area[0], area[1]])
                high = np.array([area[2], area[3]])
                state = np_random.uniform(low=low, high=high)
                for j in range(1):
                    s = state
                    env.state = state
                    for k in range(step):
                        s = clip(s, statespace_1)
                        if s[0] <= -np.pi / 6 - 0.01 or s[0] >= np.pi / 6 + 0.01:
                            unsafecount += 1
                            break
                        abstract_s = self.divide_tool.get_abstract_state(s)
                        abstract_s = str_to_list(abstract_s)
                        abstract_s = np.array(abstract_s)
                        a, logprob = self.agent.choose_action(abstract_s)
                        s_, r, done, info = env.step(a)
                        s = s_
            prob = unsafecount / 100
            # print("step=", step, "the simulated safe prob lower bound:", prob)
            return prob

        fig, ax = plt.subplots()
        prob_avg=0
        init_len = len(self.abstract_initial_states)
        prob_max=0
        norm = plt.Normalize(vmin=0.0, vmax=1)
        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=Simulate_prob(30,rect_str,[[-np.pi+0.00001, -7.9], [np.pi-0.00001, 7.9]])
            prob_avg += prob * 1.0 / init_len
            prob_max=max(prob_max,prob)
            x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
            width = x_max - x_min
            height = y_max - y_min
            # 使用概率值确定矩形的颜色
            color = plt.cm.rainbow(prob)
            # if(self.map[rect_str] == min_prob_index):
            #     min_rect_patch = patches.Rectangle((x_min, y_min), width, height,linewidth=1, edgecolor='red', facecolor=color)
            #     continue
            # color = plt.cm.viridis(prob)  # viridis 是一个颜色映射表，可以根据需要更改
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.5, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

        ax.set_xlim(region[0],region[2])
        ax.set_ylim(region[1],region[3])
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        print("The Simulated Avg:",prob_avg)
        print("The Simulated Max:",prob_max)
        plt.grid(True)
        plt.savefig(file_path)  # 保存图片
        print(f"The heatmap of Simulation have been save to{file_path}")
def VerifyFlow(args, number, seed):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    # initial_intervals = [np.pi/6,1] # level_1 abstract with 192 abs states
    #initial_intervals = [np.pi/12,0.5]#level_2 abstract with 768 abs states
    #initial_intervals = [np.pi/36,0.2]#level_3 abstract with 5760 abs states
    #initial_intervals = [np.pi / 72, 0.1]  # level_4 abstract with 23040 abs states
    #initial_intervals = [np.pi / 144, 0.05] #level5
    #initial_intervals = [np.pi / 288, 0.01]  #level6
    # initial_intervals = [np.pi / 576, 0.01]  # level7
    # initial_intervals = [np.pi / 1152, 0.01]  # level8
    initial_intervals = [np.pi / 1152, 0.005]  # level8_5
    # initial_intervals = [np.pi / 2300, 0.005]  # level9
    # initial_intervals=[np.pi / 4600, 0.005]#level10
    #initial_intervals = [np.pi / 576, 0.05]  # level_test
    state_space_1 = [[-np.pi+0.00001, -7.9], [np.pi-0.00001, 7.9]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'/home/yangjunfeng/Verify/Verify/Pendulum/rtree/level8_5')
    # train(args,number, seed, divide_tool,state_space_1,"level_7")

    # args.ActorPathLoad='/home/yangjunfeng/Verify/Verify/MDPChecking/policy_verifiability/'+ 'Pendulum_actor' + 'PD_l7_Area'+ '.pth'
    # args.CriticPathLoad='/home/yangjunfeng/Verify/Verify/MDPChecking/policy_verifiability/'+ 'Pendulum_critic' + 'PD_l7_Area'+ '.pth'
    transition = Transition_sy(args, divide_tool)
    # transition.Simulate(50, state_space_1)
    # transition.Simulate(100, state_space_1)
    # transition.Simulate(300, state_space_1)
    transition.initial_state_region = [-0.05,-0.05,0.05,0.05]
    mdp=transition.CreateMDP()
    # res = transition.ModelCheckingbyStorm(mdp, 300)
    # init_len = len(transition.abstract_initial_states)
    # ub=0
    # max_index=0
    # prob_res = 0
    # for j in range(init_len):
    #     prob_res += res[j] * 1.0 / init_len
    #     if (ub <res[j]):
    #         max_index = j
    #     ub=max(ub,res[j])
    # print("The result of Check MDP by Storm :", prob_res)
    # print("The upper bound result of Check MDP by Storm :", ub)
    # # transition.Simulate_lowerbound(300, max_index, state_space_1)
    # # transition.Simulate(120,state_space_1)
    # transition.plot_prob_area(res)
    # transition.plot_correctness()


    ver=[]
    sim=[]
    # for i in [10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]:
    for i in [6,150,300]:
        res = transition.ModelCheckingbyStorm(mdp, i)
        prob_res = 0
        init_len = len(transition.abstract_initial_states)
        ub=0
        for j in range(init_len):
            prob_res += res[j] * 1.0 / init_len
            # if (ub <res[j]):
            #     max_index = j
            ub=max(ub,res[j])
        # key = next((k for k, v in transition.map.items() if v == min_index), None)
        print("The result of Check MDP by Storm :", prob_res)
        print("The upper bound result of Check MDP by Storm :", ub)
        ver.append(ub)
        transition.Simulate(i, [-0.05,-0.05,0.05,0.05],state_space_1)
        transition.Simulate_lowerbound(i, [-0.05,-0.05,0.05,0.05], state_space_1)
    #
    # df_1 = pd.DataFrame({'V': ver, 'S': sim})
    # # df_2=pd.DataFrame({'T':transition.timecost})
    # # 将 DataFrame 保存到 Excel 文件中
    # df_1.to_excel('result.xlsx', index=False, engine='openpyxl')
    # # df_2.to_excel('time.xlsx', index=False, engine='openpyxl')

def Compare(args):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    initial_intervals = [np.pi / 1152, 0.005]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], '../Rtree/PD/level_4')

    args.ActorPathLoad="./Policies/Comparisons/Pendulum_actorlevel_8_5.pth"
    args.CriticPathLoad="./Policies/Comparisons/Pendulum_criticlevel_8_5.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [-0.05,-0.05,0.05,0.05]
    mdp=transition.CreateMDP()

    for i in [6,150,300]:
        res = transition.ModelCheckingbyStorm(mdp, i)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print(f"The Verified Avg-Bound within {i} steps:", AvgBound)
        print(f"The Verified Max-Bound within {i} steps:", Maxbound)

        Avg=transition.Simulate(i, [-0.05,-0.05,0.05,0.05], state_space_1)
        Max=transition.Simulate_lowerbound(i,[-0.05,-0.05,0.05,0.05], state_space_1)
        print(f"The Simulated Avg within {i} steps:", Avg)
        print(f"The Simulated Max within {i} steps:", Max)
        print(f"The Avg-E within {i} steps:", abs(AvgBound-Avg))
        print(f"The Max-E within {i} steps:", abs(Maxbound-Max))
def CompteAvgError(args):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    granularities = [[np.pi / 576, 0.01], [np.pi / 1152, 0.01], [np.pi / 1152, 0.005], [np.pi / 2300, 0.005], [np.pi / 4600, 0.005]]
    Levels = ["level7", "level_8", "level_8_5", "level_9", "level_10"]
    state_space_1 = [[-np.pi + 0.00001, -7.9], [np.pi - 0.00001, 7.9]]
    Policies = ["level7", "level_8", "level_8_5", "level_9", "level_10"]

    l = args.level
    if (l < 1 or l > 5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Avg-Bound Error for PD under L{l}')
        args.ActorPathLoad = './Policies/Avg-E/PD/Pendulum_actor' + Policies[l - 1] + '.pth'
        args.CriticPathLoad = './Policies/Avg-E/PD/Pendulum_critic' + Policies[l - 1] + '.pth'
        tree_path = '../Rtree/PD/' + Levels[l - 1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l - 1], [0, 1], tree_path)

        x_d = 1
        y_d = 0.8

        x_min, y_min = [-0.5,-4]
        x_max, y_max = [0.5,0]
        all_regions=[]

        all_ver=[]
        all_sim=[]
        all_error=[]
        all_time = []

        count=0
        for x in np.arange(x_min, x_max, x_d):
            for y in np.arange(y_min, y_max, y_d):
                count+=1
                box =[x+0.0001, y+0.0001, x + x_d-0.00001, y + y_d-0.00001]
                all_regions.append(box)
        print(f"A total of {count} starting regions were divided")
        # selected_rigions = random.sample(all_regions, 10)

        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            print("Starting for verification of region:",region)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for i in [10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]:
                res = transition.ModelCheckingbyStorm(mdp, i)
                init_len = len(transition.abstract_initial_states)
                AvgBound = 0
                for j in range(init_len):
                    AvgBound += res[j] * 1.0 / init_len
                print("The Verified Avg-Bound:", AvgBound)
                ver_i.append(AvgBound)
                sim_i.append(transition.Simulate(i,region,state_space_1))

            time_i = np.array(transition.timecost)
            if len(time_i) < 300:
                # 用最后一个元素填充到长度 n
                time_i = np.pad(time_i, (0, 60 - len(time_i)), 'edge')

            ver_i = np.array(ver_i)
            sim_i = np.array(sim_i)
            error_i = np.abs(ver_i - sim_i)

            all_ver.append(ver_i)
            all_sim.append(sim_i)
            all_error.append(error_i)
            all_time.append(time_i)

        all_ver = np.array(all_ver).T
        all_sim = np.array(all_sim).T
        all_error = np.array(all_error).T
        all_time = np.array(all_time).T

        mean_error = np.mean(all_error, axis=1)
        mean_time = np.mean(all_time, axis=1)

        df_validation = pd.DataFrame(all_ver, columns=[f"Box_{i + 1}_Validation" for i in range(all_ver.shape[1])])
        df_simulation = pd.DataFrame(all_sim, columns=[f"Box_{i + 1}_Simulation" for i in range(all_sim.shape[1])])
        df_error = pd.DataFrame(all_error, columns=[f"Box_{i + 1}_Error" for i in range(all_error.shape[1])])
        df_mean_error = pd.DataFrame(mean_error, columns=["Mean_Error"])

        df_time = pd.DataFrame(all_time, columns=[f"Box_{i + 1}_time" for i in range(all_time.shape[1])])
        df_mean_time = pd.DataFrame(mean_time, columns=["Mean_Time"])
        # 将验证数据、模拟数据、误差数据和平均误差合并到一个DataFrame中

        df_combined = pd.concat([df_validation, df_simulation, df_error, df_mean_error], axis=1)
        df_combined_t = pd.concat([df_time, df_mean_time], axis=1)
        # 将DataFrame保存到Excel文件
        if not os.path.exists("./results/PD"):
            os.makedirs("./results/PD")

        result_path = './results/PD/Avg_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Avg-Bound Errors at All Granularities Have Been Saved to ./results/PD/Avg_Error.xlsx')

        result_path_time = './results/PD/Timecost.xlsx'
        with pd.ExcelWriter(result_path_time, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined_t.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'TimeCost at All Granularities Have Been Saved to ./results/PD/Timecost.xlsx')

def CompteMaxError(args):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    granularities = [[np.pi / 576, 0.01], [np.pi / 1152, 0.01], [np.pi / 1152, 0.005], [np.pi / 2300, 0.005], [np.pi / 4600, 0.005]]
    Levels = ["level7", "level_8", "level_8_5", "level_9", "level_10"]
    state_space_1 = [[-np.pi + 0.00001, -7.9], [np.pi - 0.00001, 7.9]]
    Policies = ["level7", "level_8", "level_8_5", "level_9", "level_10"]

    l = args.level
    if (l < 1 or l > 5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Max-Bound Error for PD under L{l}')
        args.ActorPathLoad = './Policies/Max-E/PD/Pendulum_actor' + Policies[l - 1] + '.pth'
        args.CriticPathLoad = './Policies/Max-E/PD/Pendulum_critic' + Policies[l - 1] + '.pth'
        tree_path = '../Rtree/PD/' + Levels[l - 1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l - 1], [0, 1], tree_path)

        all_regions=[[-0.05,-0.05,0.05,0.05]]

        all_ver=[]
        all_sim=[]
        all_error=[]


        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for i in [10,20,30,40,50,60,70,80,90,100,120,150,200,250,300]:
                res = transition.ModelCheckingbyStorm(mdp, i)
                init_len = len(transition.abstract_initial_states)
                Maxbound = 0
                index = 0
                for j in range(init_len):
                    if(Maxbound<res[j]):
                        index=j
                    Maxbound = max(Maxbound, res[j])
                print("The upper bound result of Check MDP by Storm :", Maxbound)
                ver_i.append(Maxbound)
                area=transition.abstract_initial_states[index]
                area=str_to_list(area)
                sim_i.append(transition.Simulate_lowerbound(i, area,state_space_1))

            ver_i = np.array(ver_i)
            sim_i = np.array(sim_i)
            error_i = np.abs(ver_i - sim_i)

            all_ver.append(ver_i)
            all_sim.append(sim_i)
            all_error.append(error_i)

        all_ver = np.array(all_ver).T
        all_sim = np.array(all_sim).T
        all_error = np.array(all_error).T
        mean_error = np.mean(all_error, axis=1)

        df_validation = pd.DataFrame(all_ver, columns=[f"Box_{i + 1}_Validation" for i in range(all_ver.shape[1])])
        df_simulation = pd.DataFrame(all_sim, columns=[f"Box_{i + 1}_Simulation" for i in range(all_sim.shape[1])])
        df_error = pd.DataFrame(all_error, columns=[f"Box_{i + 1}_Error" for i in range(all_error.shape[1])])

        df_mean_error = pd.DataFrame(mean_error, columns=["Mean_Error"])

        # 将验证数据、模拟数据、误差数据和平均误差合并到一个DataFrame中
        df_combined = pd.concat([df_validation, df_simulation, df_error, df_mean_error], axis=1)
        # 将DataFrame保存到Excel文件
        if not os.path.exists("./results/PD"):
            os.makedirs("./results/PD")

        result_path = './results/PD/Max_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Max-Bound Errors at All Granularities Have Been Saved to ./results/PD/Max_Error.xlsx')

def Generalizability(args):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    initial_intervals = [np.pi / 576, 0.01]
    state_space_1 = [[-np.pi + 0.00001, -7.9], [np.pi - 0.00001, 7.9]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'../Rtree/PD/level_2')

    # ----------Area----------
    print("---------Start Verification for Different Initial State areas----------")
    args.ActorPathLoad = './Policies/Generalizability/Pendulum_actorPD_l7_Area.pth'
    args.CriticPathLoad = './Policies/Generalizability/Pendulum_criticPD_l7_Area.pth'
    count = 0
    regions = [[-0.55,3.50,-0.45,3.75], [0.45,-3.75,0.55,-3.50], [-0.55,0.40,-0.45,0.65]]
    for region in regions:
        print(f"Verify area- theta:[{region[0]},{region[2]}],theta_dot:[{region[1]},{region[3]}] for 150 steps")
        transition = Transition_sy(args, divide_tool)
        count += 1
        transition.initial_state_region = [region[0] + 0.0001, region[1] + 0.00001, region[2] - 0.000001,
                                           region[3] - 0.00001]
        area = [region[0] - 0.1, region[1] - 0.1, region[2] + 0.1, region[3] + 0.1]

        mdp = transition.CreateMDP()
        res = transition.ModelCheckingbyStorm(mdp, 150)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print("The Verified Avg-Bound:", AvgBound)
        print("The Verified Max-Bound:", Maxbound)

        if not os.path.exists("./plots/PD/Generalizability"):
            os.makedirs("./plots/PD/Generalizability")

        file_path = os.path.join("./plots/PD/Generalizability", f'Area{count}.pdf')
        transition.plot_prob_area(res, area, file_path)

    # #----------Horizon----------
    print("---------Start Verification for Different Horizons----------")
    transition_k = Transition_sy(args, divide_tool)
    transition_k.initial_state_region = [-0.549999,3.600001,-0.4400001,3.849999]

    area = [-0.56,3.59,-0.44,3.86]
    mdp = transition_k.CreateMDP()
    for k in [10,30,100]:
        print(f"Verify for k={k}")
        res = transition_k.ModelCheckingbyStorm(mdp, k)
        init_len = len(transition_k.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print("The Verified Avg-Bound:", AvgBound)
        print("The Verified Max-Bound:", Maxbound)
        if not os.path.exists("./plots/PD/Generalizability"):
            os.makedirs("./plots/PD/Generalizability")

        file_path = os.path.join("./plots/PD/Generalizability", f'Horizon{k}.pdf')
        transition_k.plot_prob_area(res, area, file_path)

    # -------------Policy-----------
    policies = ['PD_l7_Train2.5e4', 'PD_l7_Train1e4', 'PD_l7_Train5e4']
    for policy in policies:
        print(f"Verification of policy{policy}")
        args.ActorPathLoad = './Policies/Generalizability/' + 'Pendulum_actor' + policy + '.pth'
        args.CriticPathLoad = './Policies/Generalizability/' + 'Pendulum_critic' + policy + '.pth'
        transition = Transition_sy(args, divide_tool)
        transition.initial_state_region = [-0.549999,0.400001,-0.4400001,0.64999]
        area = [-0.56,0.39,-0.44,0.66]
        mdp = transition.CreateMDP()
        res = transition.ModelCheckingbyStorm(mdp, 120)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print("The Verified Avg-Bound:", AvgBound)
        print("The Verified Max-Bound:", Maxbound)
        if not os.path.exists("./plots/PD/Generalizability"):
            os.makedirs("./plots/PD/Generalizability")
        file_path = os.path.join("./plots/PD/Generalizability", f'{policy}.pdf')
        transition.plot_prob_area(res, area, file_path)

def Correctness(args):
    state_space = [[-np.pi, -8], [np.pi, 8]]
    initial_intervals = [np.pi / 1152, 0.01]
    state_space_1 = [[-np.pi + 0.00001, -7.9], [np.pi - 0.00001, 7.9]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'../Rtree/PD/level_8')

    args.ActorPathLoad="./Policies/Correctness/Pendulum_actorlevel_8.pth"
    args.CriticPathLoad="./Policies/Correctness/Pendulum_criticlevel_8.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [0.250001,0.150001,0.349999,0.34999]
    mdp=transition.CreateMDP()
    res = transition.ModelCheckingbyStorm(mdp, 30)
    init_len = len(transition.abstract_initial_states)
    Maxbound = 0
    AvgBound = 0
    for j in range(init_len):
        AvgBound += res[j] * 1.0 / init_len
        Maxbound = max(Maxbound, res[j])
    print("The Verified Avg-Bound:", AvgBound)
    print("The Verified Max-Bound:", Maxbound)

    if not os.path.exists("./plots/PD/Correctness"):
        os.makedirs("./plots/PD/Correctness")

    # Save results
    file_path_V = os.path.join("./plots/PD/Correctness", "PD-Verification.pdf")
    file_path_S= os.path.join("./plots/PD/Correctness", "PD-Simulation.pdf")
    plot_area=[0.25,0.15,0.35,0.35]
    transition.plot_prob_area(res,plot_area, file_path_V)
    transition.plot_correctness(plot_area,file_path_S)
if __name__ == '__main__':
    # script_path = "/home/yangjunfeng/Verify/Verify/Pendulum/policy/"
    # index_load = "level_8_5"
    # index_save = "level_8_5"
    script_path = "/home/yangjunfeng/Verify/Verify/MDPChecking/policy_verifiability/"
    index_load = "PD_l7_Train5e4"
    index_save = "PD_l7_Train5e4"

    Actor_path_load = os.path.join(script_path, "Pendulum_actor" + index_load + ".pth")
    Critic_path_load = os.path.join(script_path, "Pendulum_critic" + index_load + ".pth")
    Actor_path_save = os.path.join(script_path, "Pendulum_actor" + index_save + ".pth")
    Critic_path_save = os.path.join(script_path, "Pendulum_critic" + index_save + ".pth")

    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--envName", type=str, default="MyPendulum-v0", help="The environment")
    parser.add_argument("--ActorPathLoad", type=str, default=Actor_path_load, help="The path of file storing Actor")
    parser.add_argument("--CriticPathLoad", type=str, default=Critic_path_load, help="The path of file storing Critic")
    parser.add_argument("--ActorPathSave", type=str, default=Actor_path_save, help="The path of file storing Actor")
    parser.add_argument("--CriticPathSave", type=str, default=Critic_path_save, help="The path of file storing Critic")

    parser.add_argument("--state_dim", type=int, default=4, help="Dimension of Actor Input")
    parser.add_argument("--action_dim", type=int, default=3, help="Dimension of Actor Input")

    parser.add_argument("--max_train_steps", type=int, default=int(5e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=32,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=5e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.15, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=4, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy "
                                                                         "entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    parser.add_argument("--Experiment", type=str, default=None, help="Select Experimental Items")
    parser.add_argument("--level", type=int, default=None, help="The level of granularity")
    args = parser.parse_args()

    if(args.Experiment=="corr"):
        Correctness(args)
    elif (args.Experiment=="gen"):
        Generalizability(args)
    elif(args.Experiment=="maxe"):
        CompteMaxError(args)
    elif(args.Experiment=="avge"):
        CompteAvgError(args)
    elif(args.Experiment=="com"):
        Compare(args)