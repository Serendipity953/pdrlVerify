import pandas as pd
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import time

from Verify.AbstractTraining.ppo_discrete_main import *
from Tools.abstractor import *
from Tools.Intersector import *
from Tools.Graph import *
from queue import Queue
import numpy as np
import stormpy
import os


class Transition_sy:
    def __init__(self, args, divide_tool):
        self.agent = PPO_discrete(args)
        load(self.agent,args)
        self.action = [0, 1]
        self.initial_state = [-0.19, 3.3]  # [0.5, 0.001, 0.1, 0.001]   [0.001, 0.001, 0, 0.001]
        self.abstract_initial_states = []
        self.initial_state_region = [-3.999,3.005, 1.99999,9.99995 ]#[-0.19,3.295,-0.11,3.305]-3.99,3.005,1.99,9.995    -3.8999,3.0501,-3.8501,3.0599
        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 60
        self.divide_tool = divide_tool

        self.rtree_size = 0
        self.graph = Graph()
        # dynamics 参数
        self.a_ego = 1
        # dynamics 参数

        self.dt = .1
        self.round_rate = 100000000000000
        self.label = {}
        self.unsafe = []
        self.map={}

        self.k_steps = [10, 20, 35, 50, 70, 90, 120, 150]
        self.sim_prob = []
        self.veri_prob = []
        self.timecost=[]
    def list_to_str(self,state_list):
        obj_str = ','.join([str(_) for _ in state_list])
        return obj_str
    # 计算下一个状态可能处于的最大bound
    def next_abstract_domain(self, abstract_stat, action):  # 获取下一个状态的数据范围
        dt = self.dt  # seconds between state updates
        # current state
        delt_v0, delt_x0, delt_v1, delt_x1 = abstract_stat
        # get the value of torque from action
        if action == 0:
            acceleration = -self.a_ego
        else:
            acceleration = self.a_ego

        new_delt_x0 = delt_x0 + delt_v0 * dt
        new_delt_x1 = delt_x1 + delt_v1 * dt
        new_delt_v0 = delt_v0 - acceleration * dt
        new_delt_v1 = delt_v1 - acceleration * dt
        new_state = [new_delt_v0, new_delt_x0, new_delt_v1, new_delt_x1]
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
        dim = int(len(current)/2)
        Decision_area = self.divide_tool.intersection(current)
        for part in Decision_area:
            ith_part = str_to_list(part)
            intersection = get_intersection(dim, current, ith_part)
            flag=True
            for j in range(dim):
                if(intersection[j].upper==intersection[j].lower):
                    flag=False
            if(flag==False):
                continue
            braches.append([ith_part,ith_part])
        return braches
    def labeling(self, abstract_state):
        state_list = str_to_list(abstract_state)
        if state_list[1] > 0:
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

        #self.abstract_initial_states.append(self.divide_tool.get_abstract_state(self.initial_state))
        self.abstract_initial_states=self.divide_tool.intersection(self.initial_state_region)
        # self.abstract_initial_states.append(self.list_to_str(self.initial_state_region))
        mapping = {}
        # 初始状态添加标签
        t0 = time.time()
        print("start state:")
        state_ranges=[]
        for abstract_state in self.abstract_initial_states:
            if not mapping.__contains__(abstract_state):
                # print(abstract_state)
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
        print("start state:",)
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
        # trajectories=self.generate_trajectories(max_steps=self.limited_depth)
        # self.plot_state_ranges(state_ranges,trajectories)
        self.map=mapping
        return mdp

    def plot_state_ranges(self,state_ranges,trajectories):
        fig, ax = plt.subplots()

        # 使用PatchCollection批量绘制矩形
        patches_list = []
        for state_range in state_ranges:
            x_min,  y_min, x_max,y_max = state_range
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r',
                                     facecolor='none', linestyle='dashed')
            patches_list.append(rect)

        patch_collection = PatchCollection(patches_list, match_original=True)
        ax.add_collection(patch_collection)

        # 绘制轨迹
        for trajectory in trajectories:
            trajectory = np.array(trajectory)
            x_vals, y_vals = trajectory[:, 0], trajectory[:, 1]
            ax.plot(x_vals, y_vals, alpha=0.7)  # 设置轨迹透明度为0.5

        initial_state_low = np.array([-4, 3])
        initial_state_high = np.array([2, 10])
        plt.plot([initial_state_low[0], initial_state_high[0], initial_state_high[0], initial_state_low[0],
                  initial_state_low[0]],
                 [initial_state_low[1], initial_state_low[1], initial_state_high[1], initial_state_high[1],
                  initial_state_low[1]],
                 color='blue', linestyle='--', label='Initial State Area')
        # 设置图像显示的范围
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 30)

        # 设置坐标轴标签
        ax.set_xlabel('P')
        ax.set_ylabel('V')

        # 显示图像
        plt.grid(True)
        plt.title('State Ranges')
        plt.savefig('./ac.png')  # 保存图片
    def ModelCheckingbyStorm(self, mdp, step):
        print("-------------Verifying the  MDP in {} steps----------".format(step))
        t1 = time.time()
        formula_str = "Pmax=? [(true U<=" + str(step) + "\"unsafe\") ]"
        # formula_str = "P=? [(true U \"unsafe\") ]"
        properties = stormpy.parse_properties(formula_str)
        result = stormpy.model_checking(mdp, properties[0])
        self.veri_prob.append(1 - result.get_values()[0])
        #print("The result of Check MDP by Storm :", 1 - result.get_values()[0])
        t2 = time.time()
        print("Check the DTMC by Storm, cost time:", "[", t2 - t1, "]")
        return result.get_values()

    def generate_trajectories(self, num_episodes=50, max_steps=40):
        env = gym.make(args.envName)
        trajectories = []
        for _ in range(num_episodes):
            state = env.reset()
            trajectory = [state]
            for _ in range(max_steps):
                s = clip(state, [[-19.9999, -4.9999], [19.9999, 499.9999]])
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                next_state, reward, done, _ = env.step(a)
                trajectory.append(next_state)
                s = next_state
                if s[1] <= 0 or done:
                    break
            trajectories.append(trajectory)
        return trajectories
    def Simulate(self, step,area, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env = AdaptiveCar()
        unsafecount = 0
        np_random, seed = seeding.np_random()
        for i in range(10000):
            low = np.array([area[0], area[1]])
            high = np.array([area[2], area[3]])
            state = np_random.uniform(low=low, high=high)
            env.delta_v = state[0]
            env.delta_x = state[1]
            s=state
            for k in range(step):
                if s[1] <= 0:
                    unsafecount += 1
                    break
                s = clip(s, statespace_1)
                abstract_s = self.divide_tool.get_abstract_state(s)
                abstract_s = str_to_list(abstract_s)
                abstract_s = np.array(abstract_s)
                a, logprob = self.agent.choose_action(abstract_s)
                s_, r, done, info = env.step(a)
                s = s_
                if s[1] <= 0:
                    unsafecount += 1
                    break
        print("k=", step,"The Simulated Unsafe Prob:",  unsafecount / 10000)
        return unsafecount / 10000

    def Simulate_lowerbound(self, step,area,statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        env=AdaptiveCar()
        np_random, seed = seeding.np_random()
        max_prob=0.0
        for i in range(200):
            low = np.array([area[0], area[1]])
            high = np.array([area[2], area[3]])
            state = np_random.uniform(low=low, high=high)
            unsafecount = 0
            for j in range(200):
                s=state
                env.delta_v=state[0]
                env.delta_x = state[1]
                for k in range(step):
                    if s[1] <= 0:
                        unsafecount += 1
                        break
                    s = clip(s, statespace_1)
                    abstract_s = self.divide_tool.get_abstract_state(s)
                    abstract_s = str_to_list(abstract_s)
                    abstract_s = np.array(abstract_s)
                    a, logprob = self.agent.choose_action(abstract_s)
                    s_, r, done, info = env.step(a)
                    s = s_
                    if s[1] <= 0:
                        unsafecount += 1
                        break
            max_prob=max(max_prob,unsafecount/200)
        print("k=", step, "The Simulated Max Probability:", max_prob)
        return max_prob
    def Simulate_average(self, step, statespace_1):
        print("---------Start Simulating in {} steps---------".format(step))
        # env = gym.make(args.envName)
        env=AdaptiveCar()
        ava_prob=0.0
        for i in range(10):
            state = env.reset()
            unsafecount = 0
            for j in range(1000):
                s=state
                env.delta_v=state[0]
                env.delta_x = state[1]
                for k in range(step):
                    s = clip(s, statespace_1)
                    abstract_s = self.divide_tool.get_abstract_state(s)
                    abstract_s = str_to_list(abstract_s)
                    abstract_s = np.array(abstract_s)
                    a, logprob = self.agent.choose_action(abstract_s)
                    s_, r, done, info = env.step(a)
                    s = s_
                    if s[1] <= 0:
                        unsafecount += 1
                        break
            ava_prob+=1-(unsafecount/1000)
        ava_prob=ava_prob/10
        # print("step=", step, "the simulated safe prob average bound:", ava_prob)
        return ava_prob
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
                s_, r, done, info = env.step(1)
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

        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots()

        plt.grid(True,zorder=0)
        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=res_prob[self.map[rect_str]]
            x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
            width = x_max - x_min
            height = y_max - y_min
            # 使用概率值确定矩形的颜色
            color = plt.cm.rainbow(prob)
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=0.5, edgecolor='black', facecolor=color)
            ax.add_patch(rect)

        ax.set_xlim(area[0], area[2])
        ax.set_ylim(area[1], area[3])
        # fig.patch.set_edgecolor('#A9A9A9')
        sm = plt.cm.ScalarMappable(cmap='rainbow', norm=norm)
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=ax)
        # colorbar.outline.set_edgecolor('#A9A9A9')

        plt.savefig(file_path)  # 保存图片
        print(f"The heatmap of Verification  have been save to{file_path}")
    def plot_correctness(self,region,file_path):
        def Simulate_prob( step,state, statespace_1):
            env = AdaptiveCar()
            area = str_to_list(state)
            np_random, seed = seeding.np_random()
            unsafecount = 0
            for i in range(100):
                low = np.array([area[0], area[1]])
                high = np.array([area[2], area[3]])
                state = np_random.uniform(low=low, high=high)
                for j in range(1):
                    s = state
                    env.delta_v = state[0]
                    env.delta_x = state[1]
                    for k in range(step):
                        s = clip(s, statespace_1)
                        abstract_s = self.divide_tool.get_abstract_state(s)
                        abstract_s = str_to_list(abstract_s)
                        abstract_s = np.array(abstract_s)
                        a, logprob = self.agent.choose_action(abstract_s)
                        s_, r, done, info = env.step(a)
                        s = s_
                        if s[1] <= 0:
                            unsafecount += 1
                            break
            prob = unsafecount / 100
            return prob

        fig, ax = plt.subplots()
        prob_avg=0
        init_len = len(self.abstract_initial_states)
        prob_max=0
        norm = plt.Normalize(vmin=0.0, vmax=1)
        # 遍历矩形数据，提取坐标并绘制每个矩形
        for rect_str in self.abstract_initial_states:
            prob=Simulate_prob(30,rect_str,[[-19.9999, -4.9999], [19.9999, 499.9999]])
            prob_avg += prob * 1.0 / init_len
            prob_max=max(prob_max,prob)
            x_min, y_min, x_max, y_max = map(float, rect_str.split(','))
            width = x_max - x_min
            height = y_max - y_min
            # 使用概率值确定矩形的颜色
            color = plt.cm.rainbow(prob)
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
    state_space = [[-20, -5], [20, 500]]
    # initial_intervals = [0.1, 0.04]  # level_2 with 5050000
    # initial_intervals = [0.1, 0.02]  # level_3 with 10100000
    initial_intervals = [0.05, 0.02]  # level_4 with 20200000
    # initial_intervals = [0.05, 0.01]#level_5 with 40400000
    # initial_intervals = [0.05, 0.005]  # level_6 with 80800000
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],
                                             '/home/yangjunfeng/Verify/Verify/AdaptiveCar/rtree/level_4')
    # train(args,number, seed, divide_tool,state_space_1,"level_6")
    transition = Transition_sy(args, divide_tool)

    transition.initial_state_region = [-3.999999,3.00001,1.99999,9.99999]
    mdp=transition.CreateMDP()

    # res = transition.ModelCheckingbyStorm(mdp, 30)
    # init_len = len(transition.abstract_initial_states)
    # ub = 0
    # max_index = 0
    # prob_res = 0
    # for j in range(init_len):
    #     prob_res += res[j] * 1.0 / init_len
    #     if (ub < res[j]):
    #         max_index = j
    #     ub = max(ub, res[j])
    # print("The result of Check MDP by Storm :", prob_res)
    # print("The upper bound result of Check MDP by Storm :", ub)
    # # transition.Simulate_lowerbound(30, max_index, state_space_1)
    # # transition.Simulate(30,state_space_1)
    # transition.plot_prob_area(res)
    # transition.plot_correctness()

    ver=[]
    sim=[]
    for i in [7,30,60]:
    # for i in range(0,61,5):
        res=transition.ModelCheckingbyStorm(mdp,i)
        prob_res=0
        ub=0
        max_index=0
        init_len=len(transition.abstract_initial_states)
        for j in range(init_len):
            prob_res+=res[j]*1.0/init_len
            ub=max(ub,res[j])
        # key = next((k for k, v in transition.map.items() if v == min_index), None)
        # print(key)
        print("The result of Check MDP by Storm :", prob_res)
        print("The upper bound result of Check MDP by Storm :", ub)
        ver.append(ub)
        sim.append(transition.Simulate(i,[-3.999999,3.00001,1.99999,9.99999], state_space_1))
        transition.Simulate_lowerbound(i,[-3.999999,3.00001,1.99999,9.99999],state_space_1)

    df = pd.DataFrame({'V': ver, 'S': sim})

    # 将 DataFrame 保存到 Excel 文件中
    df.to_excel('result.xlsx', index=False, engine='openpyxl')

def Compare(args):
    state_space = [[-20, -5], [20, 500]]
    initial_intervals = [0.05, 0.02]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], '../Rtree/ACC/level_4')

    args.ActorPathLoad="./Policies/Comparisons/AdaptiveCar_actorlevel_4.pth"
    args.CriticPathLoad="./Policies/Comparisons/AdaptiveCar_criticlevel_4.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [-3.999999,3.00001,1.99999,9.99999]
    mdp=transition.CreateMDP()

    for i in [7, 30, 60]:
        res = transition.ModelCheckingbyStorm(mdp, i)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print(f"The Verified Avg-Bound within {i} steps:", AvgBound)
        print(f"The Verified Max-Bound within {i} steps:", Maxbound)

        Avg=transition.Simulate(i, [-3.999999, 3.00001, 1.99999, 9.99999], state_space_1)
        Max=transition.Simulate_lowerbound(i, [-3.999999, 3.00001, 1.99999, 9.99999], state_space_1)
        print(f"The Simulated Avg within {i} steps:", Avg)
        print(f"The Simulated Max within {i} steps:", Max)
        print(f"The Avg-E within {i} steps:", abs(AvgBound-Avg))
        print(f"The Max-E within {i} steps:", abs(Maxbound-Max) )

def CompteAvgError(args):
    state_space = [[-20, -5], [20, 500]]
    granularities=[[0.1, 0.04],[0.1, 0.04],[0.05, 0.02],[0.05, 0.01],[0.05, 0.005]]
    Levels=["level_2","level_3","level_4","level_5","level_6"]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    Policies=["level_2_avgnew","level_3_new_avg","level_4","level_5","level_6"]

    l=args.level
    if(l<1 or l>5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Avg-Bound Error for ACC under L{l}')
        args.ActorPathLoad = './Policies/Avg-E/ACC/AdaptiveCar_actor'+Policies[l-1]+'.pth'
        args.CriticPathLoad = './Policies/Avg-E/ACC/AdaptiveCar_critic'+Policies[l-1]+'.pth'
        tree_path='../Rtree/ACC/'+Levels[l-1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l-1], [0, 1], tree_path)

        x_d = -1
        y_d = 2

        x_min, y_min = [-4,4]
        x_max, y_max = [2,10]
        all_regions=[]

        all_ver=[]
        all_sim=[]
        all_error=[]
        all_time=[]

        t1=time.time()

        for r in range(1):
            box = [x_min + 0.0001, y_min + 0.0001, x_max - 0.00001, y_max - 0.00001]
            all_regions.append(box)
            y_d+=0.5
            x_min += x_d
            x_max += x_d
            y_min += y_d
            y_max += y_d+r*1.0/2+1.5


        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            print("Starting for verification of region:",region)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for i in range(0, 61, 5):
                res = transition.ModelCheckingbyStorm(mdp, i)
                init_len = len(transition.abstract_initial_states)
                AvgBound = 0
                for j in range(init_len):
                    AvgBound += res[j] * 1.0 / init_len
                print("The Verified Avg-Bound:", AvgBound)
                ver_i.append(AvgBound)
                sim_i.append(transition.Simulate(i,region,state_space_1))

            time_i = np.array(transition.timecost)
            if len(time_i) < 60:
                # 用最后一个元素填充到长度 n
                time_i = np.pad(time_i, (0, 60 - len(time_i)), 'edge')

            ver_i=np.array(ver_i)
            sim_i=np.array(sim_i)
            error_i= np.abs(ver_i - sim_i)

            all_ver.append(ver_i)
            all_sim.append(sim_i)
            all_error.append(error_i)
            all_time.append(time_i)

        all_ver = np.array(all_ver).T
        all_sim = np.array(all_sim).T
        all_error=np.array(all_error).T
        all_time=np.array(all_time).T

        mean_error = np.mean(all_error, axis=1)
        mean_time=np.mean(all_time, axis=1)

        df_validation = pd.DataFrame(all_ver, columns=[f"Box_{i+1}_Validation" for i in range(all_ver.shape[1])])
        df_simulation = pd.DataFrame(all_sim, columns=[f"Box_{i+1}_Simulation" for i in range(all_sim.shape[1])])
        df_error = pd.DataFrame(all_error,columns=[f"Box_{i + 1}_Error" for i in range(all_error.shape[1])])
        df_mean_error = pd.DataFrame(mean_error, columns=["Mean_Error"])

        df_time= pd.DataFrame(all_time, columns=[f"Box_{i + 1}_time" for i in range(all_time.shape[1])])
        df_mean_time = pd.DataFrame(mean_time, columns=["Mean_Time"])
        # 将验证数据、模拟数据、误差数据和平均误差合并到一个DataFrame中

        df_combined = pd.concat([df_validation, df_simulation, df_error, df_mean_error], axis=1)
        df_combined_t = pd.concat([df_time,df_mean_time], axis=1)
        # 将DataFrame保存到Excel文件
        if not os.path.exists("./results/ACC"):
            os.makedirs("./results/ACC")

        result_path='./results/ACC/Avg_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl',mode= 'w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Avg-Bound Errors at All Granularities Have Been Saved to ./results/ACC/Avg_Error.xlsx')

        result_path_time = './results/ACC/Timecost.xlsx'
        with pd.ExcelWriter(result_path_time, engine='openpyxl',mode= 'w') as writer:
            sheet_name = f'Level_{l}'
            df_combined_t.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'TimeCost at All Granularities Have Been Saved to ./results/ACC/Timecost.xlsx')

def CompteMaxError(args):
    state_space = [[-20, -5], [20, 500]]
    granularities=[[0.1, 0.04],[0.1, 0.04],[0.05, 0.02],[0.05, 0.01],[0.05, 0.005]]
    Levels=["level_2","level_3","level_4","level_5","level_6"]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    Policies=["level_2_avgnew","level_3_new_avg","level_4","level_5","level_6"]


    l=args.level
    if(l<1 or l>5):
        print("Get wrong level")
        return
    else:
        print(f'Start Compte the Max-Bound Error for ACC under L{l}')
        args.ActorPathLoad = './Policies/Max-E/ACC/AdaptiveCar_actor'+Policies[l-1]+'.pth'
        args.CriticPathLoad = './Policies/Max-E/ACC/AdaptiveCar_critic'+Policies[l-1]+'.pth'
        tree_path='../Rtree/ACC/'+Levels[l-1]
        divide_tool = initiate_divide_tool_rtree(state_space, granularities[l-1], [0, 1], tree_path)

        all_regions=[[-4.9999, 6.5001, 0.99999, 13.99999]]

        all_ver=[]
        all_sim=[]
        all_error=[]

        t1=time.time()

        for region in all_regions:
            transition = Transition_sy(args, divide_tool)
            ver_i=[]
            sim_i=[]
            transition.initial_state_region = region
            mdp = transition.CreateMDP()
            for k in range(0,61,5):
                res = transition.ModelCheckingbyStorm(mdp, k)
                init_len = len(transition.abstract_initial_states)
                Maxbound = 0
                AvgBound = 0
                for j in range(init_len):
                    AvgBound += res[j] * 1.0 / init_len
                    Maxbound = max(Maxbound, res[j])
                print("The Verified Avg-Bound:", AvgBound)
                print("The Verified Max-Bound:", Maxbound)
                ver_i.append(Maxbound)
                area=region
                sim_i.append(transition.Simulate_lowerbound(k, area,state_space_1))

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
        if not os.path.exists("./results/ACC"):
            os.makedirs("./results/ACC")

        result_path='./results/ACC/Max_Error.xlsx'
        with pd.ExcelWriter(result_path, engine='openpyxl',mode= 'w') as writer:
            sheet_name = f'Level_{l}'
            df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f'Max-Bound Errors at All Granularities Have Been Saved to ./results/ACC/Max_Error.xlsx')

def Generalizability(args):
    state_space = [[-20, -5], [20, 500]]
    initial_intervals = [0.1, 0.04]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'../Rtree/ACC/level_2')


    # #----------Area----------
    print("---------Start Verification for Different Initial State areas----------")
    args.ActorPathLoad='./Policies/Generalizability/AdaptiveCar_actorAC_l2_Area.pth'
    args.CriticPathLoad='./Policies/Generalizability/AdaptiveCar_criticAC_l2_Area.pth'
    count = 0
    # regions=[[-4,9,-2,10],[-3,6,-1,7],[-2,4,0,5]]
    regions=[[-5,9,-3,10],[-2.1,6,-0.1,7],[-0.7,4,1.3,5]]
    for region in regions:
        print(f"Verify area- delta_v:[{region[0]},{region[2]}],delta_x:[{region[1]},{region[3]}] for 40 steps")
        transition = Transition_sy(args, divide_tool)
        count+=1
        transition.initial_state_region = [region[0]+0.0001,region[1]+0.00001,region[2]-0.000001,region[3]-0.00001]
        area=[region[0]-0.1,region[1]-0.1,region[2]+0.1,region[3]+0.1]

        mdp = transition.CreateMDP()
        res = transition.ModelCheckingbyStorm(mdp, 40)
        init_len = len(transition.abstract_initial_states)
        Maxbound = 0
        AvgBound = 0
        for j in range(init_len):
            AvgBound += res[j] * 1.0 / init_len
            Maxbound = max(Maxbound, res[j])
        print("The Verified Avg-Bound:", AvgBound)
        print("The Verified Max-Bound:", Maxbound)

        if not os.path.exists("./plots/ACC/Generalizability"):
            os.makedirs("./plots/ACC/Generalizability")

        file_path = os.path.join("./plots/ACC/Generalizability", f'Area{count}.eps')
        transition.plot_prob_area(res,area,file_path)


    #
    #
    # # #----------Horizon----------
    # print("---------Start Verification for Different Horizons----------")
    # transition_k = Transition_sy(args, divide_tool)
    # transition_k.initial_state_region = [-2.99999,4.00001,-1.00001,4.99999]
    #
    # area=[-3.1,3.9,-0.9,5.1]
    # mdp = transition_k.CreateMDP()
    # for k in [20,25,30]:
    #     print(f"Verify for k={k}")
    #     res = transition_k.ModelCheckingbyStorm(mdp, k)
    #     init_len = len(transition_k.abstract_initial_states)
    #     Maxbound = 0
    #     AvgBound = 0
    #     for j in range(init_len):
    #         AvgBound += res[j] * 1.0 / init_len
    #         Maxbound = max(Maxbound, res[j])
    #     print("The Verified Avg-Bound:", AvgBound)
    #     print("The Verified Max-Bound:", Maxbound)
    #     if not os.path.exists("./plots/ACC/Generalizability"):
    #         os.makedirs("./plots/ACC/Generalizability")
    #
    #     file_path = os.path.join("./plots/ACC/Generalizability", f'Horizon{k}.pdf')
    #     transition_k.plot_prob_area(res, area, file_path)

    # -------------Policy-----------
    # policies=['AC_l2_1e5','AC_l2_3e5','AC_l2_5e5']
    # for policy in policies:
    #     print(f"Verification of policy{policy}")
    #     args.ActorPathLoad='./Policies/Generalizability/'+ 'AdaptiveCar_actor' + policy + '.pth'
    #     args.CriticPathLoad='./Policies/Generalizability/'+ 'AdaptiveCar_critic' + policy + '.pth'
    #     transition = Transition_sy(args, divide_tool)
    #     transition.initial_state_region = [-3.49999,4.00001,-1.50001,4.99999]
    #     area=[-3.6,3.9,-1.4,5.1]
    #     mdp = transition.CreateMDP()
    #     res = transition.ModelCheckingbyStorm(mdp, 30)
    #     init_len = len(transition.abstract_initial_states)
    #     Maxbound = 0
    #     AvgBound = 0
    #     for j in range(init_len):
    #         AvgBound += res[j] * 1.0 / init_len
    #         Maxbound = max(Maxbound, res[j])
    #     print("The Verified Avg-Bound:", AvgBound)
    #     print("The Verified Max-Bound:", Maxbound)
    #     if not os.path.exists("./plots/ACC/Generalizability"):
    #         os.makedirs("./plots/ACC/Generalizability")
    #     file_path = os.path.join("./plots/ACC/Generalizability", f'{policy}.pdf')
    #     transition.plot_prob_area(res, area, file_path)

def Correctness(args):
    state_space = [[-20, -5], [20, 500]]
    initial_intervals = [0.05, 0.02]
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],'../Rtree/ACC/level_4')

    args.ActorPathLoad="./Policies/Correctness/AdaptiveCar_actorlevel_4.pth"
    args.CriticPathLoad="./Policies/Correctness/AdaptiveCar_criticlevel_4.pth"

    transition = Transition_sy(args, divide_tool)
    transition.initial_state_region = [-4.999999,8.00001,-3.00001,8.49999]
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

    if not os.path.exists("./plots/ACC/Correctness"):
        os.makedirs("./plots/ACC/Correctness")

    # Save results
    file_path_V = os.path.join("./plots/ACC/Correctness", "ACC-Verification.pdf")
    file_path_S= os.path.join("./plots/ACC/Correctness", "ACC-Simulation.pdf")
    plot_area=[-5.1,7.9,-2.9,8.6]
    transition.plot_prob_area(res,plot_area, file_path_V)
    transition.plot_correctness(plot_area,file_path_S)

if __name__ == '__main__':
    # script_path = "/home/yangjunfeng/Verify/Verify/AdaptiveCar/policy/"
    # script_path = "/home/yangjunfeng/Verify/Verify/MDPChecking/policy_error/Max_E/"
    script_path = "/home/yangjunfeng/Verify/Verify/MDPChecking/policy_verifiability/"
    index = "level_2"
    # index_load = "level_2_vary"
    # index_save = "level_2_vary"
    index_load = "AC_l2_5e5"
    index_save = "AC_l2_5e5"

    # Actor_path = os.path.join(script_path, "AdaptiveCar_actor" + index + ".pth")
    # Critic_path = os.path.join(script_path, "AdaptiveCar_critic" + index + ".pth")

    Actor_path_load = os.path.join(script_path, "AdaptiveCar_actor" + index_load + ".pth")
    Critic_path_load = os.path.join(script_path, "AdaptiveCar_critic" + index_load + ".pth")
    Actor_path_save = os.path.join(script_path, "AdaptiveCar_actor" + index_save + ".pth")
    Critic_path_save = os.path.join(script_path, "AdaptiveCar_critic" + index_save + ".pth")


    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--envName", type=str, default="MyAdaptiveCar-v0", help="The environment")
    parser.add_argument("--ActorPathLoad", type=str, default=Actor_path_load, help="The path of file storing Actor")
    parser.add_argument("--CriticPathLoad", type=str, default=Critic_path_load, help="The path of file storing Critic")
    parser.add_argument("--ActorPathSave", type=str, default=Actor_path_save, help="The path of file storing Actor")
    parser.add_argument("--CriticPathSave", type=str, default=Critic_path_save, help="The path of file storing Critic")
    # parser.add_argument("--ActorPath", type=str, default=Actor_path, help="The path of file storing Actor")
    # parser.add_argument("--CriticPath", type=str, default=Critic_path, help="The path of file storing Critic")

    parser.add_argument("--state_dim", type=int, default=4, help="Dimension of Actor Input")
    parser.add_argument("--action_dim", type=int, default=2, help="Dimension of Actor Input")

    parser.add_argument("--max_train_steps", type=int, default=int(1e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e2,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
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
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
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

