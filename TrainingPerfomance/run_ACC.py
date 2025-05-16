import argparse
import os
import random

import numpy as np

from scipy.interpolate import griddata

from ppo_discrete_main_abs import *
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D



# import seaborn as sns

def Setabs_parser(index, script_path):
    # index = "level_3_5_bad"
    envName = "MyAdaptiveCar-v0"
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--envName", type=str, default=envName, help="The environment")
    Actor_path = os.path.join(script_path, "BouncingBall_actor" + index + ".pth")
    Critic_path = os.path.join(script_path, "BouncingBall_critic" + index + ".pth")

    parser.add_argument("--ActorPath", type=str, default=Actor_path, help="The path of file storing Actor")
    parser.add_argument("--CriticPath", type=str, default=Critic_path, help="The path of file storing Critic")

    parser.add_argument("--state_dim", type=int, default=4, help="Dimension of Actor Input")
    parser.add_argument("--action_dim", type=int, default=2, help="Dimension of Critic Input")

    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help=" Maximum number of training steps")
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

    return parser


def Setparser(script_path):
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    envName = "MyAdaptiveCar-v0"
    parser.add_argument("--envName", type=str, default=envName, help="The environment")
    Actor_path = os.path.join(script_path, envName + "_actor" + ".pth")
    Critic_path = os.path.join(script_path, envName + "_critic" + ".pth")

    parser.add_argument("--ActorPath", type=str, default=Actor_path, help="The path of file storing Actor")
    parser.add_argument("--CriticPath", type=str, default=Critic_path, help="The path of file storing Critic")

    parser.add_argument("--state_dim", type=int, default=2, help="Dimension of Actor Input")
    parser.add_argument("--action_dim", type=int, default=2, help="Dimension of Critic Input")

    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help=" Maximum number of training steps")
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

    return parser



if __name__ == '__main__':
    abs_script_path = "./abs_policy/"
    script_path = "./abs_policy/policy/"

    state_space = [[-20, -5], [20, 500]]
    initial_intervals = [0.1, 0.04]  # level_2 with 5050000
    # initial_intervals = [0.1, 0.02]  # level_3 with 10100000
    # initial_intervals = [0.05, 0.02]  # level_4 with 20200000
    # initial_intervals = [0.05, 0.01]#level_5 with 40400000
    # initial_intervals = [0.05, 0.005]  # level_6 with 80800000
    state_space_1 = [[-19.9999, -4.9999], [19.9999, 499.9999]]
    divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1],
                                             '../Rtree/ACC/level_2')
    index = "level_2"
    # #traditional training and get the model
    parser = Setparser(script_path)
    args = parser.parse_args()
    abs_parser = Setabs_parser(index, abs_script_path)
    abs_args = abs_parser.parse_args()

    args.max_episode_steps=1000
    abs_args.max_episode_steps=1000
    num_trainings = 5
    num_episodes = 600
    total_rewards_std = np.zeros((num_trainings, num_episodes))
    total_rewards_abs = np.zeros((num_trainings, num_episodes))
    for i in range(num_trainings):
        rewards_abs = abs_train(abs_args, 1, 50, divide_tool, state_space_1, index)
        rewards_std = train(args,1,50,state_space_1)
        total_rewards_std[i] = rewards_std
        total_rewards_abs[i] = rewards_abs

    mean_rewards_std = total_rewards_std.mean(axis=0)
    std_rewards_std = total_rewards_std.std(axis=0)

    mean_rewards_abs = total_rewards_abs.mean(axis=0)
    std_rewards_abs = total_rewards_abs.std(axis=0)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_rewards_std, label='TT',color='#2c9576')
    plt.fill_between(range(num_episodes), mean_rewards_std - std_rewards_std, mean_rewards_std + std_rewards_std, color='#2c9576',
                     alpha=0.3)

    plt.plot(mean_rewards_abs, label='AT',color='#d05c5c')
    plt.fill_between(range(num_episodes), mean_rewards_abs - std_rewards_abs, mean_rewards_abs + std_rewards_abs, color='#d05c5c',
                     alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)  # 主刻度
    plt.xlabel('Episodes',fontsize=18)
    # plt.ylabel('Cumulative Reward',fontsize=18)
    plt.legend(fontsize=14)
    plt.savefig('./trainplot/ACC.svg')  # 保存图片
    # abstract training and get the model



