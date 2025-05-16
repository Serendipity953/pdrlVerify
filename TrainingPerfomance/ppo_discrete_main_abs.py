import torch
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
from matplotlib import pyplot as plt
from Tools.abstractor import str_to_list, initiate_divide_tool_rtree
import pandas as pd
from openpyxl.reader.excel import load_workbook
from env.adaptivecar import *
from env.bouncingball import *
from env.pendulum_v1 import *
def clip(state,statespace_1):
    # low = state_space[0][0]
    # high = state_space[1][0]
    for i in range(len(statespace_1[0])):
        state[i] = np.clip(state[i], statespace_1[0][i], statespace_1[1][i])
    return state
def save(agent,args):
    torch.save(agent.actor.state_dict(), args.ActorPath)
    torch.save(agent.critic.state_dict(), args.CriticPath)
    print("---for task:",args.envName,'|save model to:',args.ActorPath)

def load(agent,args):
    agent.actor.load_state_dict(torch.load(args.ActorPath))
    agent.critic.load_state_dict(torch.load(args.CriticPath))
    print("---for task:",args.envName,'|load model from:',args.ActorPath)
def evaluate_policy(args, env, agent):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        # if args.use_state_norm:  # During the evaluating,update=False
        #     s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        step=0
        while not done and step<=args.max_episode_steps:
            a, prob = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
            step+=1
        evaluate_reward += episode_reward

    return evaluate_reward / times


def train(args,number, seed,statespace_1):
    if (args.envName == "MyAdaptiveCar-v0"):
        env = AdaptiveCar()
        env_evaluate = AdaptiveCar()
        env.mode = 2
        env_evaluate.mode = 2
    elif (args.envName == "MyBouncingball-v0"):
        env = BouncingBall()
        env_evaluate = BouncingBall()
        env.mode = 0
        env_evaluate.mod=0
    elif (args.envName == "MyPendulum-v0"):
        env = Pendulum()
        env_evaluate = Pendulum()
        env.mode = 0
        env_evaluate.mode = 0
    # env = gym.make(args.envName)
    # env_evaluate = gym.make(args.envName)  # When evaluating the policy, we need to rebuild an environment
    #Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    #args.state_dim = env.observation_space.shape[0]
    #args.action_dim = env.action_space.n
    # args.max_episode_steps=500
    #args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.envName))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    mean_rewards = []#Mean rewards of recent 10 evaluate
    ep=[]## Record the step length of evaluations, using to plot

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    #load(agent,args)
    # Build a tensorboard
    #writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done and episode_steps< args.max_episode_steps:
            s = clip(s, statespace_1)
            episode_steps += 1
            #print(episode_steps)
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                ep.append(evaluate_num)
                evaluate_reward = evaluate_policy(args, env_evaluate, agent)
                if (evaluate_reward < -2000):
                    evaluate_rewards.append(-2000)
                else:
                    evaluate_rewards.append(evaluate_reward)
                mean_rewards.append(torch.mean(torch.Tensor(evaluate_rewards[-10:])))
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                #writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                #if evaluate_num % args.save_freq == 0:
                    #np.save('./data_train/PPO_discrete_env_{}_number_{}_seed_{}.npy'.format(env_name, number, seed), np.array(evaluate_rewards))

    return evaluate_rewards

def abs_evaluate_policy(args, env, agent,state_norm,divide_tool,statespace_1):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        #transfer the state to abstracted state
        done = False
        episode_reward = 0
        step=0
        while not done and step<=args.max_episode_steps:
            s=clip(s,statespace_1)

            abstract_s = divide_tool.get_abstract_state(s)
            abstract_s = str_to_list(abstract_s)
            abstract_s = np.array(abstract_s)
            if args.use_state_norm:  # During the evaluating,update=False
                abstract_s = state_norm(abstract_s, update=False)
            a,prob = agent.choose_action(abstract_s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
            step+=1
        evaluate_reward += episode_reward

    return evaluate_reward / times


def abs_train(args,  number, seed, divide_tool,statespace_1,index):
    if (args.envName == "MyAdaptiveCar-v0"):
        env = AdaptiveCar()
        env_evaluate = AdaptiveCar()
        env.mode = 2
        env_evaluate.mode = 2
    elif (args.envName == "MyBouncingball-v0"):
        env = BouncingBall()
        env_evaluate = BouncingBall()
        env.mode = 0
        env_evaluate.mod=0
    elif (args.envName == "MyPendulum-v0"):
        env = Pendulum()
        env_evaluate = Pendulum()
        env.mode = 0
        env_evaluate.mode = 0
    # env = gym.make(args.envName)
    # env_evaluate = gym.make(args.envName)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    #args.state_dim = env.observation_space.shape[0]*2
    #args.action_dim = env.action_space.n
    # args.max_episode_steps = 500
    # args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.envName))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    mean_rewards = []  # Mean rewards of recent 10 evaluate
    ep = []  ## Record the step length of evaluations, using to plot

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)
    # load(agent,args)
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_discrete/env_{}_number_{}_seed_{}'.format(env_name, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    do = True
    while total_steps < args.max_train_steps and do == True:
        s = env.reset()

        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done and episode_steps < args.max_episode_steps:
            episode_steps += 1
            # print(episode_steps)
            s = clip(s, statespace_1)
            abstract_s = divide_tool.get_abstract_state(s)
            abstract_s = str_to_list(abstract_s)
            abstract_s = np.array(abstract_s)
            if args.use_state_norm:
                abstract_s = state_norm(abstract_s)
            a, a_logprob = agent.choose_action(abstract_s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            s_ = clip(s_, statespace_1)

            abstract_s_ = divide_tool.get_abstract_state(s_)
            abstract_s_ = str_to_list(abstract_s_)
            abstract_s_ = np.array(abstract_s_)
            if args.use_state_norm:
                abstract_s_ = state_norm(abstract_s_)

            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(abstract_s, a, a_logprob, r, abstract_s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                ep.append(evaluate_num)
                evaluate_reward = abs_evaluate_policy(args, env_evaluate, agent,state_norm, divide_tool, statespace_1)
                if(evaluate_reward<-2000):
                    evaluate_rewards.append(-2000)
                else:
                    evaluate_rewards.append(evaluate_reward)
                mean_rewards.append(torch.mean(torch.Tensor(evaluate_rewards[-10:])))
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                # if torch.mean(torch.Tensor(evaluate_rewards[-10:]))>-200:
                # do=False
                # print("reached finishing condition")
                # break
    return evaluate_rewards
    # save(agent,args)
