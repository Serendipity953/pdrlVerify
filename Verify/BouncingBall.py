import string

import pandas as pd
from matplotlib import patches, cm
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
        load(self.agent, args)
        self.action = [0, 1]
        self.initial_state = [4.603, 0.715]
        self.abstract_initial_states = []
        # self.initial_state_region = [5,-0.1,9,0]
        self.initial_state_region = [3.0001, -10.399899999999992, 3.99999, -8.800009999999991]
        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000

        self.limited_depth = 26
        self.divide_tool = divide_tool

        self.rtree_size = 0
        self.graph = Graph()
        # dynamics 参数
        self.dt = 0.1

        self.g = 9.81
        self.round_rate = 100000000000000

        self.timecost = []
        self.label = {}
        self.unsafe = []
        self.map={}
        self.k_steps = [5, 10, 15, 20, 25, 30, 40, 60, 100]

    def list_to_str(self, state_list):
        obj_str = ','.join([str(_) for _ in state_list])
        return obj_str

    # 计算下一个状态可能处于的最大bound
    def first_stage_domain(self, abstract_stat):
        p_0, v_0, p_1, v_1 = abstract_stat
        dt = self.dt
        g = self.g
        # standard operation
        v_0_second = v_0 - 9.81 * dt
        p_0_second = max(p_0 + dt * v_0_second, -0.99999)
        v_1_second = v_1 - 9.81 * dt
        p_1_second = max(p_1 + dt * v_1_second, -0.99999)

        first_domain = [p_0_second, v_0_second, p_1_second, v_1_second]
        return first_domain
    def get_abs(self,state,S,D):
        abs=[]
        low=[]
        high=[]
        for i in range(len(state)):
            low.append(S[i]+D[i]*math.floor((state[i]-S[i])/D[i]))
            high.append(S[i] + D[i] * math.floor((state[i] - S[i]+D[i]) / D[i]))
        for item in low:
            abs.append(item)
        for item in high:
            abs.append(item)
        return abs
    def get_next_states(self, decision_part, transition_part, case):
        # the return:successors with correspond prob
        suc_with_prob = []
        p_0, v_0, p_1, v_1 = transition_part
        curS = torch.unsqueeze(torch.tensor(decision_part, dtype=torch.float), 0)
        prob = self.agent.actor(curS).detach().numpy().flatten()
        # case0:action is irrelevant :v<=0 && p<=0
        if (case == 0):
            v_0_prime = -0.9 * v_1
            v_1_prime = -0.9 * v_0
            p_0_prime = 0
            p_1_prime = 0.001
            successor = [p_0_prime, v_0_prime, p_1_prime, v_1_prime]
            suc_with_prob.append([successor, 1.0])
        # case3:the ball out of reach and not bounce,action is irrelevant,keep origin state
        elif (case == 3):
            suc_with_prob.append([transition_part, 1.0])
        # case1 or case2: compute the prob of taking each action over the states
        elif (case == 1):
            # case1 take action 0:
            suc_with_prob.append([transition_part, prob[0]])
            # suc_with_prob.append([transition_part, 1])
            # case1 take action 1:
            v_0_prime = v_0 - 4
            v_1_prime = v_1 - 4
            p_0_prime = 3.999
            p_1_prime = 4
            successor = [p_0_prime, v_0_prime, p_1_prime, v_1_prime]
            suc_with_prob.append([successor, prob[1]])
            # suc_with_prob.append([successor, 1.0])
        elif (case == 2):
            # case2 take action 0:
            suc_with_prob.append([transition_part, prob[0]])
            # suc_with_prob.append([transition_part, 1])
            # case2 take action 1:
            v_0_prime = -0.9 * v_1 - 4
            v_1_prime = -0.9 * v_0 - 4
            p_0_prime = 3.999
            p_1_prime = 4
            successor = [p_0_prime, v_0_prime, p_1_prime, v_1_prime]
            suc_with_prob.append([successor, prob[1]])
            # suc_with_prob.append([successor, 1.0])
        # compute the successors under each action

        return suc_with_prob

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
    #         p_0, v_0, p_1, v_1 =transition_area
    #         if (p_1 <= 0 and v_1<=0):
    #             case=0
    #             braches.append([ith_part, transition_area, case])
    #         elif (p_0>=4 and v_1<=0):
    #             case=1
    #             braches.append([ith_part, transition_area, case])
    #         elif (p_0>=4 and v_0>=0):
    #             case=2
    #             braches.append([ith_part, transition_area, case])
    #         elif(p_0>=0 and p_1<=4):
    #             case=3
    #             braches.append([ith_part,transition_area,case])
    #     return braches
    def refine_branch(self, first_domain):
        # return: branchs in MDP,each branch has a decision part and an area to compute the successors
        braches = []
        Decision_area = self.divide_tool.intersection(first_domain)
        for part in Decision_area:
            ith_part = str_to_list(part)
            p_0, v_0, p_1, v_1 = ith_part
            if (p_1 <= 0 and v_1 <= 0):
                case = 0
                braches.append([ith_part, ith_part, case])
            elif (p_0 >= 4 and v_1 <= 0):
                case = 1
                braches.append([ith_part, ith_part, case])
            elif (p_0 >= 4 and v_0 >= 0):
                case = 2
                braches.append([ith_part, ith_part, case])
            elif (p_0 >= 0 and p_1 <= 4):
                case = 3
                braches.append([ith_part, ith_part, case])
        return braches

    def labeling(self, abstract_state):
        state_list = str_to_list(abstract_state)
        if not (state_list[0] < 1 and (
                (state_list[1] < -7 and state_list[3] > -7) or (state_list[1] > -7 and state_list[3] < 7) or (
                state_list[1] < 7 and state_list[3] > 7))):
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
        print("Steps :", 0, "   |   ", "Transition Size :", abstract_state_count, "   |   ", "Time Cost:", 0)
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
                if self.label[mapping[self.list_to_str(current)]] == "unsafe":
                    builder.new_row_group(row)
                    builder.add_next_value(row, mapping[self.list_to_str(current)], 1)
                    row += 1
                    continue
                first_domain = self.first_stage_domain(current)
                current_branches = self.refine_branch(first_domain)
                builder.new_row_group(row)  # current states as new row group, each row in this group is a branch
                for j in range(len(current_branches)):
                    successor = self.get_next_states(current_branches[j][0], current_branches[j][1],
                                                     current_branches[j][2])
                    for k in range(len(successor)):
                        # if the successor is not visited and safe, give it a state index, and add it to the bfs queue
                        successor_str = self.list_to_str(successor[k][0])
                        if not mapping.__contains__(successor_str):
                            mapping[successor_str] = abstract_state_count
                            self.label[mapping[successor_str]] = self.labeling(successor_str)
                            if self.label[mapping[successor_str]] == 'unsafe':
                                self.unsafe.append(mapping[successor_str])
                            abstract_state_count += 1
                            bfs.put(successor[k][0])
                            state_ranges.append(successor[k][0])
                        # 添加该分支下的后续迁移，但是注意不保证在该分支下，两个动作迁移至同一个状态，可能有bug
                        builder.add_next_value(row, mapping[successor_str], successor[k][1])
                    row += 1
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
        self.map = mapping
        # trajectories=self.generate_trajectories(max_steps=self.limited_depth)
        # self.plot_state_ranges(state_ranges,trajectories)
        return mdp

    def ModelCheckingbyStorm(self, mdp, step):
        print("-------------Verifying the  MDP in {} steps----------".format(step))
        t1 = time.time()
        formula_str = "Pmax=? [(true U<=" + str(step) + "\"unsafe\") ]"
        # formula_str = "P=? [(true U \"unsafe\") ]"
        properties = stormpy.parse_properties(formula_str)
        result = stormpy.model_checking(mdp, properties[0])
        # print("The result of Check MDP by Storm :", 1 - result.get_values()[0])
        t2 = time.time()
        print("Check the MDP by Storm, cost time:", "[", t2 - t1, "]")
        return result.get_values()

    def plot_state_ranges(self, state_ranges, trajectories):
        # print(trajectories)
        fig, ax = plt.subplots(figsize=(42, 80))
        # 使用PatchCollection批量绘制矩形
        patches_list = []
        for i, state_range in enumerate(state_ranges):
            x_min, y_min, x_max, y_max = state_range
            alpha = (i + 1) / len(state_ranges)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='red',
                                     facecolor='none', linestyle='dashed', alpha=alpha)
            patches_list.append(rect)

        patch_collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(patch_collection)
        # 绘制轨迹
        for trajectory in trajectories:
            trajectory = np.array(trajectory)
            x_vals, y_vals = trajectory[:, 0], trajectory[:, 1]
            ax.scatter(x_vals, y_vals, alpha=1, s=10)  # 设置轨迹透明度为0.5
            ax.plot(x_vals, y_vals, alpha=0.5)  # 设置轨迹透明度为0.5

        initial_state_low = np.array([5, -1])
        initial_state_high = np.array([9, 1])
        plt.plot([initial_state_low[0], initial_state_high[0], initial_state_high[0], initial_state_low[0],
                  initial_state_low[0]],
                 [initial_state_low[1], initial_state_low[1], initial_state_high[1], initial_state_high[1],
                  initial_state_low[1]],
                 color='blue', linestyle='--', label='Initial State Area')

        # unsafe 区域
        unsafe_low = np.array([-1, -7])
        unsafe_high = np.array([1, 7])
        plt.plot([unsafe_low[0], unsafe_high[0], unsafe_high[0], unsafe_low[0],
                  unsafe_low[0]],
                 [unsafe_low[1], unsafe_low[1], unsafe_high[1], unsafe_high[1],
                  unsafe_low[1]],
                 color='red', linestyle='--', label='Unsafe Area')
        # 设置图像显示的范围
        ax.set_xlim(-1, 15)
        ax.set_ylim(-20, 20)

        # 设置坐标轴标签
        ax.set_xlabel('Position', fontsize=50)
        ax.set_ylabel('Velocity', fontsize=50)
        ax.tick_params(axis='both', which='major', labelsize=40)
        plt.legend(fontsize=40)
        # 显示图像
        plt.grid(True)
        # plt.title('State Ranges')
        plt.savefig('./bb4.png')  # 保存图片

    def generate_trajectories(self, num_episodes=40, max_steps=60):
        # env = gym.make(args.envName)
        env = BouncingBall()
        env.mode = 2
        trajectories = []
        unsafe = 0
        for _ in range(num_episodes):
            state = env.reset()
            s = state
            trajectory = [state]
            for _ in range(max_steps):
                s = clip(s, [[-0.9999, -19.9999], [19.9999, 19.9999]])
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                next_state, reward, done, _ = env.step(a)
                trajectory.append(next_state)
                s = next_state
                if s[0] < 1 and s[1] > -7 and s[1] < 7:
                    unsafe += 1
                    break
            trajectories.append(trajectory)
        print(unsafe)
        return trajectories

    def Simulate(self, step, area,statespace_1,gran):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env = BouncingBall()
        np_random, seed = seeding.np_random()
        env.mode = 2
        # env_evaluate = gym.make(args.envName)  # When evaluating the policy, we need to rebuild an environment
        # Set random seed
        unsafecount = 0
        for i in range(10000):
            low = np.array([area[0], area[1]])
            high = np.array([area[2], area[3]])
            state = np_random.uniform(low=low, high=high)
            env.p = state[0]
            env.v = state[1]
            for k in range(step):
                s = np.array((env.p, env.v))
                if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                    unsafecount += 1
                    break
                env.v = env.v - 9.81 * self.dt
                env.p = max(env.p + self.dt * env.v, 0)
                s = np.array((env.p, env.v))
                s = clip(s, statespace_1)
                abstract_s=self.get_abs(s,[-1, -20],gran)
                # abstract_s = self.divide_tool.get_abstract_state(s)
                # abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                s_, r, done, info = env.step(a)
                # s = s_
                if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                    unsafecount += 1
                    break
        print("K=", step, "The Simulated Unsafe Prob:",  unsafecount / 10000)
        return  unsafecount / 10000

    def Simulate_average(self, step, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env = BouncingBall()
        env.mode = 2
        ava_prob = 0.0
        np_random, seed = seeding.np_random()
        for i in range(20):
            # state = env.reset()
            low = np.array([7,-0.2])
            high = np.array([8, 0.2])
            state = np_random.uniform(low=low, high=high)
            unsafecount = 0
            # print(i)
            for j in range(1000):
                s = state
                env.p = state[0]
                env.v = state[1]
                for k in range(step):
                    s = clip(s, statespace_1)
                    if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                        unsafecount += 1
                        break
                    abstract_s = self.divide_tool.get_abstract_state(s)
                    abstract_s = str_to_list(abstract_s)
                    abstract_s = np.array(abstract_s)
                    a, logprob = self.agent.choose_action(abstract_s)
                    s_, r, done, info = env.step(a)
                    s = s_
            ava_prob += (unsafecount / 1000)
        ava_prob = ava_prob / 20
        print("step=", step, "the simulated safe prob average bound:", ava_prob)
        return ava_prob

    def test(self):
        env = gym.make(args.envName)
        maxp = 0
        maxv = 0
        minp = 0
        minv = 0
        for i in range(10000):
            s = env.reset()
            for k in range(400):
                maxp = max(maxp, s[0])
                minp = min(minp, s[0])
                maxv = max(maxv, s[1])
                minv = min(minv, s[1])
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                s_, r, done, info = env.step(a)
                s = s_
                if done:
                    break
        print(minp, minv, maxp, maxv)

    def Simulate_lowerbound(self, step,area, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        env = BouncingBall()
        max_prob = 0.0
        np_random, seed = seeding.np_random()
        for i in range(100):
            low = np.array([area[0], area[1]])
            high = np.array([area[2], area[3]])
            state = np_random.uniform(low=low, high=high)
            unsafecount = 0
            for j in range(100):
                # s = state
                env.p = state[0]
                env.v = state[1]
                for k in range(step):
                    s = np.array((env.p, env.v))
                    if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                        unsafecount += 1
                        break
                    env.v = env.v - 9.81 * self.dt
                    env.p = max(env.p + self.dt * env.v, 0)
                    s = np.array((env.p, env.v))
                    s = clip(s, statespace_1)
                    abstract_s = self.divide_tool.get_abstract_state(s)
                    abstract_s = str_to_list(abstract_s)
                    abstract_s = np.array(abstract_s)
                    a, logprob = self.agent.choose_action(abstract_s)
                    s_, r, done, info = env.step(a)
                    s = s_
                    if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                        unsafecount += 1
                        break
            max_prob = max(max_prob, unsafecount / 100)
        print("step=", step, "the simulated unsafe prob upper bound:", max_prob)
        return max_prob
    def plot_prob_area(self,res_prob,area,file_path):
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

        fig,ax = plt.subplots()

        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=res_prob[self.map[rect_str]]
            x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
            width = x_max - x_min
            height = y_max - y_min
            # 使用概率值确定矩形的颜色
            color_idx = int(norm(prob) * (num_colors - 1))
            color = plt.cm.rainbow(prob)
            # if(self.map[rect_str] == min_prob_index):
            #     min_rect_patch = patches.Rectangle((x_min, y_min), width, height,linewidth=1, edgecolor='red', facecolor=color)
            #     continue
            # color = plt.cm.viridis(prob)  # viridis 是一个颜色映射表，可以根据需要更改
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.2, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
        # for rect_str in self.abstract_initial_states:
        #     if (self.map[rect_str] == min_prob_index):
        #         prob = res_prob[self.map[rect_str]]
        #         x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
        #         width = x_max - x_min
        #         height = y_max - y_min
        #         # 使用概率值确定矩形的颜色
        #         color_idx = int(norm(prob) * (num_colors - 1))
        #         color = plt.cm.rainbow(prob)
        #         # color = gradient_colors[color_idx]
        #         min_rect_patch = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='red',facecolor=color)
        #         ax.add_patch(min_rect_patch)
        #         break

        ax.set_xlim(area[0], area[2])
        ax.set_ylim(area[1], area[3])
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)

        # cbar.ax.invert_yaxis()  # Reverse the colorbar direction to match the gradient

        # ax.text(0.5, 1.02, f"Avg Probability Bound: {avg_prob:.2f}", transform=ax.transAxes, fontsize=12,
        #         verticalalignment='bottom', horizontalalignment='center')
        # ax.text(0.5, 1.05, f"Worst Probability Bound: {min_prob:.2f}", transform=ax.transAxes, fontsize=12,
        #         verticalalignment='bottom', horizontalalignment='center')

        plt.grid(True)
        plt.savefig(file_path)  # 保存图片
        print(f"The heatmap of Verification  have been save to{file_path}")
    def plot_correctness(self,region,file_path):
        def Simulate_prob( step,state, statespace_1):
            print("---------Start Simulating in {} steps---------".format(step))
            # env = gym.make(args.envName)
            env = BouncingBall()
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
                    # s = state
                    env.p = state[0]
                    env.v = state[1]
                    for k in range(step):
                        env.v = env.v - 9.81 * self.dt
                        env.p = max(env.p + self.dt * env.v, 0)
                        s = np.array((env.p, env.v))
                        s = clip(s, statespace_1)
                        abstract_s = self.divide_tool.get_abstract_state(s)
                        abstract_s = str_to_list(abstract_s)
                        abstract_s = np.array(abstract_s)
                        a, logprob = self.agent.choose_action(abstract_s)
                        s_, r, done, info = env.step(a)
                        s = s_
                        if (s[0] < 1 and s[1] > -7 and s[1] < 7):
                            unsafecount += 1
                            break
            prob = unsafecount / 100
            # print("step=", step, "the simulated safe prob lower bound:", prob)
            return prob

        fig, ax = plt.subplots()
        prob_avg=0
        init_len = len(self.abstract_initial_states)
        prob_max=0
        max_area=""
        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=Simulate_prob(26,rect_str,[[-0.9999, -19.9999], [19.9999, 19.9999]])
            prob_avg += prob * 1.0 / init_len
            if(prob_max<prob):
                max_area=rect_str
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
        norm = plt.Normalize(vmin=0.0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)

        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        print("The Simulated Avg:",prob_avg)
        print("The Simulated Max:",prob_max)

        plt.grid(True)
        plt.savefig(file_path)  # 保存图片
        print(f"The heatmap of Simulation have been save to{file_path}")
    def plot_color_bar(self):
        import matplotlib as mpl
        fig = plt.figure(figsize=(6, 1.5))  # 控制宽高比，水平放置并苗条
        ax = fig.add_axes([0.1, 0.5, 0.8, 0.3])  # 控制颜色条的位置和大小
        norm = plt.Normalize(vmin=0.0, vmax=1)

        # 创建一个只显示颜色条的图形


        # 创建并添加颜色条
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax, orientation='vertical')

        # 隐藏坐标轴（如果有多余的轴）
        ax.tick_params(left=False, labelleft=False)
        cbar.ax.tick_params(labelsize=10)
        # 显示颜色条
        if not os.path.exists("./plots/BB/paper"):
            os.makedirs("./plots/BB/paper")

            # 保存图片
        file_path = os.path.join("./plots/BB/paper", "colorbar.eps")
        # plt.grid(True)
        plt.savefig(file_path)  # 保存图片



def VerifyFlow(args, number, seed):
    # create the asbtract state sapce by rtree
    state_space = [[-1, -20], [20, 20]]
    # initial_intervals = [0.5,0.5]#level_1
    # initial_intervals = [0.05, 0.05]  # level_2
    # initial_intervals = [0.01, 0.01]  # level_3
    # initial_intervals = [0.005, 0.01]  # level_4 1680,0000
    # initial_intervals = [0.005, 0.008]  # level_4_5 2100,0000
    # initial_intervals = [0.005, 0.005]  # level_5 3360,0000
    initial_intervals = [0.004, 0.005]  # level_6 4200,0000
    # initial_intervals = [0.004, 0.004]  # level_7 4200,0000
    state_space_1 = [[-0.9999, -19.9999], [19.9999, 19.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],
                                             '/home/yangjunfeng/Verify/Verify/BouncingBall/rtree/level_6')
    # teststep(state_space_1, divide_tool)
    train(args, number, seed, divide_tool, state_space_1, "level_4_5")
    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [5.00001, -0.99999, 8.9999, 0.999999]
    # transition.Simulate(10, state_space_1)
    # transition.Simulate(30, state_space_1)
    # transition.Simulate(600, state_space_1)
    # transition.test()

    mdp = transition.CreateMDP()

    # res = transition.ModelCheckingbyStorm(mdp, 26)
    # init_len = len(transition.abstract_initial_states)
    # ub = 0
    # max_index = 0
    # prob_res = 0
    # for j in range(init_len):
    #     prob_res += res[j] * 1.0 / init_len
    #     if (ub< res[j]):
    #         max_index = j
    #     ub = max(ub, res[j])
    # print("The result of Check MDP by Storm :", prob_res)
    # print("The upper bound result of Check MDP by Storm :", ub)
    # # transition.Simulate_lowerbound(28, 0, state_space_1)
    # # transition.Simulate(26,0,state_space_1)
    # # transition.Simulate_average(27,state_space_1)
    # transition.plot_prob_area(res)
    # transition.plot_correctness()
    # # # transition.plot_color_bar()

    ver = []
    sim = []
    # for i in range(20,61,5):
    for i in [20,40,60]:
        res = transition.ModelCheckingbyStorm(mdp, i)
        prob_res = 0
        init_len = len(transition.abstract_initial_states)
        ub=0
        index=0
        for j in range(init_len):
            prob_res += res[j] * 1.0 / init_len
            # if (ub< res[j]):
            #     index = j
            ub = max(ub, res[j])
        print("The result of Check MDP by Storm :", prob_res)
        ver.append(prob_res)
        print("The upper bound result of Check MDP by Storm :", ub)
        # sim.append(transition.Simulate_average(i, state_space_1))
        transition.Simulate_lowerbound(i, [5.00001, -0.99999, 8.9999, 0.999999],state_space_1)
        # print(index)
        sim.append(transition.Simulate(i,[5.00001, -0.99999, 8.9999, 0.999999], state_space_1,initial_intervals))

    # trajectories = transition.generate_trajectories(max_steps=transition.limited_depth)
    # transition.plot_state_ranges([], trajectories)
    # df = pd.DataFrame({'V': ver, 'S': sim, 'T': transition.timecost})
    df = pd.DataFrame({'V': ver, 'S': sim})

    # 将 DataFrame 保存到 Excel 文件中
    df.to_excel('result.xlsx', index=False, engine='openpyxl')
def Compare(args):
    state_space = [[-1, -20], [20, 20]]
    initial_intervals = [0.004, 0.005]
    state_space_1 = [[-0.9999, -19.9999], [19.9999, 19.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], '../Rtree/BB/level_6')

    args.ActorPathLoad="./Policies/Comparisons/BouncingBall_actorlevel_6_com.pth"
    args.CriticPathLoad="./Policies/Comparisons/BouncingBall_criticlevel_6_com.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [5.00001, -0.99999, 8.9999, 0.999999]
    mdp=transition.CreateMDP()

    for i in [20]:
        res = transition.ModelCheckingbyStorm(mdp, i)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print(f"The Verified Avg-Bound within {i} steps:", AvgBound)
        print(f"The Verified Max-Bound within {i} steps:", Maxbound)

        Avg=transition.Simulate(i, [5.00001, -0.99999, 8.9999, 0.999999], state_space_1,initial_intervals)
        Max=transition.Simulate_lowerbound(i, [5.00001, -0.99999, 8.9999, 0.999999], state_space_1)
        print(f"The Simulated Avg within {i} steps:", Avg)
        print(f"The Simulated Max within {i} steps:", Max)
        print(f"The Avg-E within {i} steps:", abs(AvgBound-Avg))
        print(f"The Max-E within {i} steps:", abs(Maxbound-Max) )
def CompteAvgError(args):
    state_space = [[-1, -20], [20, 20]]
    granularities=[[0.005, 0.01],[0.005, 0.008],[0.005, 0.005] ,[0.004, 0.005],[0.004, 0.004]]
    Levels=["level_4","level_4_5","level_5","level_6","level_7"]
    state_space_1 = [[-0.9999, -19.9999], [19.9999, 19.9999]]
    Policies=["level_4","level_4_5","level_5","level_6","level_7"]

    l = args.level
    if (l < 1 or l > 5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Avg-Bound Error for BB under L{l}')
        args.ActorPathLoad = './Policies/Avg-E/BB/BouncingBall_actor' + Policies[l - 1] + '.pth'
        args.CriticPathLoad = './Policies/Avg-E/BB/BouncingBall_critic' + Policies[l - 1] + '.pth'
        tree_path = '../Rtree/BB/' + Levels[l - 1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l - 1], [0, 1], tree_path)

        all_regions=[[14.0001, 2.00001, 14.99999, 3.99999],[5.0001,-0.99999,6.99999,-0.00001],[3.0001,-2.999999, 4.99999, -2.0001],[1.0001,-7.999999, 2.99999, -7.0001],[6.0001, 2.00001, 7.99999, 2.9999999987]]#
        all_ver=[]
        all_sim=[]
        all_error=[]
        all_time = []



        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            print("Starting for verification of region:",region)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for i in range(0, 61,5):
                res = transition.ModelCheckingbyStorm(mdp, i)
                init_len = len(transition.abstract_initial_states)
                AvgBound = 0
                for j in range(init_len):
                    AvgBound += res[j] * 1.0 / init_len
                print("The Verified Avg-Bound:", AvgBound)
                ver_i.append(AvgBound)
                sim_i.append(transition.Simulate(i,region,state_space_1,granularities[l - 1]))

            time_i=np.array(transition.timecost)
            if len(time_i) < 60:
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
        if not os.path.exists("./results/BB"):
            os.makedirs("./results/BB")

        result_path = './results/BB/Avg_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Avg-Bound Errors at All Granularities Have Been Saved to ./results/BB/Avg_Error.xlsx')

        result_path_time = './results/BB/Timecost.xlsx'
        with pd.ExcelWriter(result_path_time, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined_t.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'TimeCost at All Granularities Have Been Saved to ./results/BB/Timecost.xlsx')

def CompteMaxError(args):
    state_space = [[-1, -20], [20, 20]]
    granularities=[[0.005, 0.01],[0.005, 0.008],[0.005, 0.005] ,[0.004, 0.005],[0.004, 0.004]]
    Levels=["level_4","level_4_5","level_5","level_6","level_7"]
    state_space_1 = [[-0.9999, -19.9999], [19.9999, 19.9999]]
    Policies=["level_4","level_4_5","level_5","level_6","level_7"]

    l = args.level
    if (l < 1 or l > 5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Max-Bound Error for BB under L{l}')
        args.ActorPathLoad = './Policies/Max-E/BB/BouncingBall_actor' + Policies[l - 1] + '.pth'
        args.CriticPathLoad = './Policies/Max-E/BB/BouncingBall_critic' + Policies[l - 1] + '.pth'
        tree_path = '../Rtree/BB/' + Levels[l - 1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l - 1], [0, 1], tree_path)

        all_regions=[[7.00001,-0.19999,7.9999,0.199999]]

        all_ver=[]
        all_sim=[]
        all_error=[]


        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for i in range(0, 61, 5):
                res = transition.ModelCheckingbyStorm(mdp, i)
                init_len = len(transition.abstract_initial_states)
                Maxbound = 0
                AvgBound = 0
                for j in range(init_len):
                    AvgBound += res[j] * 1.0 / init_len
                    Maxbound = max(Maxbound, res[j])
                print("The Verified Avg-Bound:", AvgBound)
                print("The Verified Max-Bound:", Maxbound)
                ver_i.append(Maxbound)
                sim_i.append(transition.Simulate_lowerbound(i, region,state_space_1))

            ver_i=np.array(ver_i)
            sim_i=np.array(sim_i)
            error_i= np.abs(ver_i - sim_i)

            all_ver.append(ver_i)
            all_sim.append(sim_i)
            all_error.append(error_i)


        all_ver = np.array(all_ver).T
        all_sim = np.array(all_sim).T
        all_error=np.array(all_error).T
        mean_error = np.mean(all_error, axis=1)

        df_validation = pd.DataFrame(all_ver, columns=[f"Box_{i+1}_Validation" for i in range(all_ver.shape[1])])
        df_simulation = pd.DataFrame(all_sim, columns=[f"Box_{i+1}_Simulation" for i in range(all_sim.shape[1])])
        df_error = pd.DataFrame(all_error,columns=[f"Box_{i + 1}_Error" for i in range(all_error.shape[1])])

        df_mean_error = pd.DataFrame(mean_error, columns=["Mean_Error"])

        # 将验证数据、模拟数据、误差数据和平均误差合并到一个DataFrame中
        df_combined = pd.concat([df_validation, df_simulation, df_error, df_mean_error], axis=1)
        # 将DataFrame保存到Excel文件
        if not os.path.exists("./results/BB"):
            os.makedirs("./results/BB")

        result_path = './results/BB/Max_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl', mode='w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Max-Bound Errors at All Granularities Have Been Saved to ./results/BB/Max_Error.xlsx')

def Correctness(args):
    state_space = [[-1, -20], [20, 20]]
    initial_intervals = [0.004, 0.005]
    state_space_1 = [[-0.9999, -19.9999], [19.9999, 19.9999]]
    # divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'../Rtree/BB/level_6')
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],
                                             '/home/yangjunfeng/Verify/Verify/BouncingBall/rtree/level_6')
    args.ActorPathLoad="./Policies/Correctness/BouncingBall_actorlevel_6.pth"
    args.CriticPathLoad="./Policies/Correctness/BouncingBall_criticlevel_6.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region =  [5.600001, 0.700001, 5.699999, 0.949999]
    mdp=transition.CreateMDP()
    res = transition.ModelCheckingbyStorm(mdp, 26)
    init_len = len(transition.abstract_initial_states)
    Maxbound = 0
    AvgBound = 0
    for j in range(init_len):
        AvgBound += res[j] * 1.0 / init_len
        Maxbound = max(Maxbound, res[j])
    print("The Verified Avg-Bound:", AvgBound)
    print("The Verified Max-Bound:", Maxbound)

    if not os.path.exists("./plots/BB/Correctness"):
        os.makedirs("./plots/BB/Correctness")

    # Save results
    file_path_V = os.path.join("./plots/BB/Correctness", "BB-Verification.pdf")
    file_path_S= os.path.join("./plots/BB/Correctness", "BB-Simulation.pdf")
    plot_area= [-5.6,0.7,5.7,0.95]
    transition.plot_prob_area(res,plot_area, file_path_V)
    transition.plot_correctness(plot_area,file_path_S)
if __name__ == '__main__':
    script_path = "/home/yangjunfeng/Verify/Verify/BouncingBall/policy/Avg-E"
    # script_path = "/home/yangjunfeng/Verify/Verify/MDPChecking/policy_error/Max_E/"
    # script_path = "/home/yangjunfeng/Verify/Verify/BouncingBall/policy/"

    index = "level_6"
    # Actor_path = os.path.join(script_path, "BouncingBall_actor" + index + ".pth")
    # Critic_path = os.path.join(script_path, "BouncingBall_critic" + index + ".pth")

    index_load = "level_6"
    index_save = "level_6"

    Actor_path_load = os.path.join(script_path, "BouncingBall_actor" + index_load + ".pth")
    Critic_path_load = os.path.join(script_path, "BouncingBall_critic" + index_load + ".pth")
    Actor_path_save = os.path.join(script_path, "BouncingBall_actor" + index_save + ".pth")
    Critic_path_save = os.path.join(script_path, "BouncingBall_critic" + index_save + ".pth")


    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--envName", type=str, default="MyBouncingball-v0", help="The environment")
    parser.add_argument("--ActorPathLoad", type=str, default=Actor_path_load, help="The path of file storing Actor")
    parser.add_argument("--CriticPathLoad", type=str, default=Critic_path_load, help="The path of file storing Critic")
    parser.add_argument("--ActorPathSave", type=str, default=Actor_path_save, help="The path of file storing Actor")
    parser.add_argument("--CriticPathSave", type=str, default=Critic_path_save, help="The path of file storing Critic")


    # parser.add_argument("--ActorPath", type=str, default=Actor_path, help="The path of file storing Actor")
    # parser.add_argument("--CriticPath", type=str, default=Critic_path, help="The path of file storing Critic")

    parser.add_argument("--state_dim", type=int, default=4, help="Dimension of Actor Input")
    parser.add_argument("--action_dim", type=int, default=2, help="Dimension of Critic Input")

    parser.add_argument("--max_train_steps", type=int, default=int(5e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=32,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=5e-3, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-3, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.15, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=16, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    parser.add_argument("--Experiment", type=str, default=None, help="Select Experimental Items")
    parser.add_argument("--level", type=int, default=None, help="The level of granularity")
    args = parser.parse_args()
    args.Experiment = "corr"

    if(args.Experiment=="corr"):
        Correctness(args)
    elif(args.Experiment=="maxe"):
        CompteMaxError(args)
    elif(args.Experiment=="avge"):
        CompteAvgError(args)
    elif(args.Experiment=="com"):
        Compare(args)

