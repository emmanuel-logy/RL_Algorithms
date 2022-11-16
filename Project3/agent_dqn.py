#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import signal
import time
import logging
import shelve

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Housekeeping
        self.writer = SummaryWriter('/home/ejayaraju/workspace/WPI-CS525-DS595-Fall22/Project3/runs/P3_trySam2')   # default replay_buff to disk"
        signal.signal(signal.SIGINT, self.exit_gracefully)      # to write replay_buff to disk 
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        self.log_filepath = "/home/ejayaraju/workspace/WPI-CS525-DS595-Fall22/Project3/train_outputs/9_trySam2/progress.logs"

        # RL related parameters
        self.env = env
        self.nA = env.action_space.n
        self.epsilon = 1.0
        self.epsilon_lowerbound = 0.025
        self.total_train_episodes = 100000  # To make epsilon decay linearly to 2.5% only towards the end 
        self.episodes_for_epsilon_decay = 40000
        self.episodes = 0
        self.avg_over_episodes = 30
        self.gamma = 0.99
        self.skip_frames = 1               # i.e. skip none.. After talking to prof.. maybe the good frames where ball hits the base correctly is lost
        self.algo = 'dqn'                 # or 'ddqn'

        # Replay Buffer related parameters
        self.replay_buff_filepath = "/home/ejayaraju/workspace/WPI-CS525-DS595-Fall22/Project3/replay_buffer.pickle"
        self.replay_buff_fileHandle = None
        self.init_replay_buff_size = 3000
        self.buf_size = 30000   # more is always good for DL
        self.buffer = deque(maxlen=self.buf_size)

        if args.train_dqn:
            self.replay_buff_fileHandle = shelve.open(self.replay_buff_filepath)
            if 'replay_buffer' in self.replay_buff_fileHandle:
                self.buffer = self.replay_buff_fileHandle['replay_buffer']
        
        # DL related parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device configuration
        self.batch_size = 32                # David Silver lecture says 64
        self.learning_rate = 1e-4
        self.train_steps = 1
        self.Qpolicy_update_interval = 5000    # instead of num_epochs = 5
        if args.test_dqn:
            self.checkpt_filepath = "./DQN_checkpoint.pth_best"
        else:
            self.checkpt_filepath = "/home/ejayaraju/workspace/WPI-CS525-DS595-Fall22/Project3/train_outputs/9_trySam2/DQN_checkpoint.pth"
        self.Qpolicy = DQN(in_channels=4, num_actions=self.nA).float().to(device=self.device)
        self.Qnet = DQN(in_channels=4, num_actions=self.nA).float().to(device=self.device)

        self.optimizer = optim.Adam(self.Qnet.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()
        
        if os.path.exists(self.checkpt_filepath):
            checkpoint = torch.load(self.checkpt_filepath, map_location=self.device)
            self.Qpolicy.load_state_dict(checkpoint["Qpolicy_net_state"])
            self.Qnet.load_state_dict(checkpoint["Qnet_state"])
            self.optimizer.load_state_dict(checkpoint["optim_state"])
            self.epsilon = checkpoint["epsilon"]
            self.episodes = checkpoint["episodes"]
            self.train_steps = checkpoint["training_steps"]
            # self.episodes = 22440+1
            # self.train_steps = 2328020+1

        if args.train_dqn:
            self.Qnet.train()
            self.Qpolicy.train()
            # call init functions
            self.init_replay_buff()
            self.logger = self.init_logger()
        
        elif args.test_dqn:
        # else:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.Qnet.eval()
            self.Qpolicy.eval()



    def init_logger(self):
        logging.basicConfig(filename=self.log_filepath,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
        return logging.getLogger('DQN_Logger')


    def exit_gracefully(self, sig, frame):
        print('\n You pressed Ctrl+C!')
        print('saving stuff . . . ')
        self.save_current_model()
        print('saving stuff Done!!')
        sys.exit(0)


    def numpy_to_torch(self, s):
        s = np.swapaxes(s,0,2)              # To convert 4*84*84 into 84*84*4
        s = np.transpose(s, (0, 2, 1))      # To rotate 84*84 img clockwise to view properly
        s = s.astype(np.float32)
        s = s/255
        s = torch.from_numpy(s)
        return s


    def init_replay_buff(self):
        """
        Initialize replay buffer B with atleast batch_size samples before starting 
        the DQN training loop, which populates the buffer as the training goes along
        """
        while len(self.buffer) <= self.init_replay_buff_size:
            # Generate (s,a,r,s') and append to buffer
            frames = 0
            s = self.env.reset()
            s = self.numpy_to_torch(s)
            
            # To display images on Tensorboard
            # for i in range(len(s)):
            #     img = torch.from_numpy(s[i].reshape(1,84,84))
            #     self.writer.add_image('observation '+str(i), img)            
            #     self.writer.flush()
            # self.writer.close()
            
            # Till the end of episode
            while True:
                # get an action from policy... make_action(Q,s,nA,epsilon)
                a = self.make_action(s)

                # return a new state, reward and done
                s_next, r, terminated, truncated, info = self.env.step(a)    # r --> reward
                frames += 1
                if terminated or truncated:
                    # if s_next is None, Qpolicy(s_next) cannot be computed .. so no use!
                    # s_next = None
                    # buff_elem = (s,a,r,s_next)
                    # self.buffer.append(buff_elem)
                    break

                # Populate replay buffer
                s_next = self.numpy_to_torch(s_next)
                if frames % self.skip_frames == 0:
                    buff_elem = (s,a,r,s_next)
                    self.buffer.append(buff_elem)
                
                # update state
                s = s_next 
        
        self.save_replay_buffer()


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, s, test=False):
        """
        Return predicted action of your agent
        Input:
            s: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # define decaying epsilon
        # if args.test_dqn:
        #     self.epsilon = self.epsilon_lowerbound
        # % 10 ..just to reduce epsilon every 10 steps.. so that gets clipped at 2.5% around 2300th training step
        # if self.train_steps % 10 == 0:
        #     if self.epsilon > self.epsilon_lowerbound:
        #         self.epsilon = max(self.epsilon_lowerbound, 0.99*self.epsilon)
        # moved to train loop to update once every episode is over

        with torch.no_grad():
            if test:        
                s = self.numpy_to_torch(s)
                s = s.to(device=self.device)
                Q_sa = self.Qpolicy(s)
                action = torch.argmax(Q_sa).item()    # greedy_action_index
            else:
                # choose an action using epsilon-greedy logic
                s = s.to(device=self.device)
                if self.algo == 'dqn':
                    Q_sa = self.Qnet(s)    # Qnet returns 4 values which are Q(s,a1), Q(s,a2) Q(s,a3), Q(s,a4)
                    Q_avg = Q_sa
                elif self.algo == 'ddqn':
                    Q_sa = self.Qnet(s)  
                    Qpolicy_sa = self.Qpolicy(s)
                    Q_avg = (Q_sa + Qpolicy_sa)/2
                # greedy_action_index = torch.argmax(Q_avg)
                # prob = [self.epsilon/self.nA] * self.nA
                # prob[greedy_action_index.item()] = 1 - self.epsilon + (self.epsilon/self.nA)
                # action = np.random.choice(range(self.nA), p=prob)
                probability_list = np.ones([self.nA,1])*(self.epsilon / self.nA)
                q, argq = Q_avg.data.to(self.device).max(1)
                probability_list[argq[0].item()] += 1 - self.epsilon
                action = torch.tensor([np.random.choice(np.arange(self.nA), p=probability_list.flatten())]).item()
            ###########################
        
        return action
    
    def push(self):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        
        
        ###########################
        return 
    
    def save_replay_buffer(self):
        self.replay_buff_fileHandle['replay_buffer'] = self.buffer
        self.replay_buff_fileHandle.sync()


    def save_current_model(self, is_best=False):
        checkpoint = {
                        "epsilon": self.epsilon,
                        "episodes": self.episodes,
                        "training_steps": self.train_steps,
                        "optim_state": self.optimizer.state_dict(),
                        "Qpolicy_net_state": self.Qpolicy.state_dict(),
                        "Qnet_state": self.Qnet.state_dict()
                    }
        if is_best:
            torch.save(checkpoint, self.checkpt_filepath+'_best')
        else:
            torch.save(checkpoint, self.checkpt_filepath)


    def get_train_sars(self):
        mini_batch = deque(maxlen=self.batch_size)
        rand_indices = np.random.randint(low=0, high=len(self.buffer), size=self.batch_size)
        for i in rand_indices:
            mini_batch.append(self.buffer[i])
        return mini_batch
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        """
        [0] Generate some 5000 states in replay buffer B initially
        Training Loop:        
        [1] Take action 'a' according to epsilon-greedy policy
        [2] Store (s,a,r,s') in replay buffer B
        [3] Sample a mini-batch from B uniformly
        [4] Copmute Qt(s,a) from target network with w_fixed
        [6] Compute loss = [r + g * max Qt(s,a) - Q(s,a) ]^2 
        [7] Optimize using grad descent
        """

        total_reward = 0.0
        avg_reward = 0.0
        best_avg_reward = 0.0
        
        print (f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Epsilon: {self.epsilon}')
        self.logger.info(f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Epsilon: {self.epsilon}')
        
        # while avg_reward < 20.0:
        while self.episodes <= self.total_train_episodes:

            # Linear Decay Formula: f(t) = C - r*t
            if self.epsilon > self.epsilon_lowerbound:
                self.epsilon = 1.0 - self.episodes*(1/self.episodes_for_epsilon_decay)

            frames = 0
            self.episodes += 1  
            s = self.env.reset()
            s = self.numpy_to_torch(s)

            # Till the end of episode
            while True:
                # get an action from policy... make_action(Q,s,nA,epsilon)
                a = self.make_action(s)
                # print(">>>>>>> Action: ", a, " <<<<<<<")

                # return a new state, reward and done
                s_next, r, terminated, truncated, info = self.env.step(a)    # r --> reward
                total_reward += r
                frames += 1
                if terminated or truncated:
                    # if s_next is None, Qpolicy(s_next) cannot be computed .. so no use!
                    # s_next = None
                    # buff_elem = (s,a,r,s_next)
                    # self.buffer.append(buff_elem)
                    break

                # Populate replay buffer
                s_next = self.numpy_to_torch(s_next)
                if frames % self.skip_frames == 0:
                    buff_elem = (s,a,r,s_next)
                    self.buffer.append(buff_elem)
            
                ### TD update tables ###
                # td_target = r + self.gamma * max(self.Qpolicy(s_next))   # td_target with best Q
                # td_error = td_target - self.Qnet(s)[a]                   # td_error
                
                ###  TD update deepRL ###
                self.train_steps += 1

                if self.train_steps % 10 == 0:
                    # Sample mini-batch from B uniformly
                    mini_batch = self.get_train_sars()  # it uses self.batch_size
                    
                    # tensor of 64 train data ... batch_size * 4*84*84
                    s_batch = torch.empty(self.batch_size, s.shape[0], s.shape[1], s.shape[2])
                    a_batch = torch.empty(self.batch_size)
                    r_batch = torch.empty(self.batch_size)
                    s_next_batch = torch.empty(self.batch_size, s.shape[0], s.shape[1], s.shape[2])

                    # tuple (s,a,r,s')...sars[0]-->s; sars[3]-->s'; so on...
                    for i, sars in enumerate(mini_batch):
                        s_batch[i] = sars[0]     
                        a_batch[i] = sars[1]
                        r_batch[i] = sars[2]
                        s_next_batch[i] = sars[3]
                        
                    # td_target --> [r + gamma * Qpolicy(s',a')]
                    r_batch = r_batch.to(device=self.device)
                    s_next_batch = s_next_batch.to(device=self.device)
                    td_target = 0.0
                    with torch.no_grad():
                        Qpolicy_val = self.Qpolicy(s_next_batch)            # batch_size * 4 is ouput of network
                        Qpolicy_val, ind = torch.max(Qpolicy_val, dim=1)   # we need just max of each row
                        td_target = r_batch + self.gamma * Qpolicy_val

                    # cur_Qnet_val --> Qnet(s,a)
                    s_batch = s_batch.to(device=self.device)
                    Qnet_val = self.Qnet(s_batch)
                    Qnet_sa = torch.empty(self.batch_size)
                    for i, qval in enumerate(Qnet_val):
                        Qnet_sa[i] = qval[int(a_batch[i].item())]

                    # td_error --> Loss = [td_target - Qnet(s,a)]^2
                    td_target = td_target.to(device=self.device)
                    Qnet_sa = Qnet_sa.to(device=self.device)
                    loss = self.criterion(Qnet_sa, td_target)
                    
                    # update Qnet   ## Q[s][a] = Q[s][a] + alpha * (td_error).. alpha is weight gradients
                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.Qnet.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()
        
                # update state
                s = s_next 

                # plot on tensorboard 
                self.writer.add_scalar('Training_Steps vs Epsilon', self.epsilon, self.train_steps)

                # save progress to disk
                if self.train_steps % self.Qpolicy_update_interval == 0:
                    self.Qpolicy.load_state_dict(self.Qnet.state_dict())
                    self.save_current_model()
                    
            if self.episodes % 20000 == 0:
                self.save_replay_buffer()

            # save progress to disk and print
            if self.episodes % self.avg_over_episodes == 0:
                avg_reward = total_reward / self.avg_over_episodes
                total_reward = 0
                print (f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Loss: {loss.item():.4f}, Epsilon: {self.epsilon}')
                self.logger.info(f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Loss: {loss.item():.4f}, Epsilon: {self.epsilon}')

                # plot on tensorboard
                # writer.add_scalar(label, y, x))
                self.writer.add_scalar('Episodes vs Epsilon', self.epsilon, self.episodes)
                self.writer.add_scalar('Episodes vs Avg_Reward', avg_reward, self.episodes)
                self.writer.add_scalar('Training_Steps vs Avg_Reward', avg_reward, self.train_steps)
                self.writer.add_scalar('Training_Steps vs Loss', loss, self.train_steps)
                self.writer.add_scalar('Training_Steps vs Episodes', self.episodes, self.train_steps)
                
                best_avg_reward = max(best_avg_reward, avg_reward)
                if best_avg_reward > avg_reward:
                    self.save_current_model(is_best=True)

        ###########################



    def train_ddqn(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        """
        [0] Generate some 5000 states in replay buffer B initially
        Training Loop:        
        [1] Take action 'a' according to epsilon-greedy policy
        [2] Store (s,a,r,s') in replay buffer B
        [3] Sample a mini-batch from B uniformly
        [4] Copmute Qt(s,a) from target network with w_fixed
        [6] Compute loss = [r + g * max Qt(s,a) - Q(s,a) ]^2 
        [7] Optimize using grad descent
        """

        total_reward = 0.0
        avg_reward = 0.0
        
        print (f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Epsilon: {self.epsilon}')
        self.logger.info(f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Epsilon: {self.epsilon}')
        
        # while self.episodes <= self.total_train_episodes:
                        
        while avg_reward < 20.0:
            # print (f'Episode [{self.episodes}], Step [{self.train_steps}], Total_Reward: [{total_reward}]')
            
            # Linear Decay Formula: f(t) = C - r*t
            if self.epsilon > self.epsilon_lowerbound:
                self.epsilon = 1.0 - self.episodes*(1/self.total_train_episodes)

            frames = 0
            self.episodes += 1  
            s = self.env.reset()
            s = self.numpy_to_torch(s)

            # Till the end of episode
            while True:
                # Get an action
                a = self.make_action(s, algo='ddqn')

                # return a new state, reward and done
                s_next, r, terminated, truncated, info = self.env.step(a)    # r --> reward
                total_reward += r
                frames += 1
                if terminated or truncated:
                    # if s_next is None, Qpolicy(s_next) cannot be computed .. so no use!
                    # s_next = None
                    # buff_elem = (s,a,r,s_next)
                    # self.buffer.append(buff_elem)
                    break

                # Populate replay buffer
                s_next = self.numpy_to_torch(s_next)
                if frames % self.skip_frames == 0:
                    buff_elem = (s,a,r,s_next)
                    self.buffer.append(buff_elem)
            
                ### TD update tables ###
                # td_target = r + self.gamma * max(self.Qpolicy(s_next))   # td_target with best Q
                # td_error = td_target - self.Qnet(s)[a]                   # td_error
                
                ###  TD update deepRL ###
                # Sample mini-batch from B uniformly
                self.train_steps += 1
                mini_batch = self.get_train_sars()  # it uses self.batch_size
                
                # tensor of 64 train data ... batch_size * 4*84*84
                s_batch = torch.empty(self.batch_size, s.shape[0], s.shape[1], s.shape[2])
                a_batch = torch.empty(self.batch_size)
                r_batch = torch.empty(self.batch_size)
                s_next_batch = torch.empty(self.batch_size, s.shape[0], s.shape[1], s.shape[2])

                # tuple (s,a,r,s')...sars[0]-->s; sars[3]-->s'; so on...
                for i, sars in enumerate(mini_batch):
                    s_batch[i] = sars[0]     
                    a_batch[i] = sars[1]
                    r_batch[i] = sars[2]
                    s_next_batch[i] = sars[3]
                    
                r_batch = r_batch.to(device=self.device)
                s_next_batch = s_next_batch.to(device=self.device)

                # ddqn changes
                opt = np.random.choice([0,1], p=[0.5,0.5])

                # td_error --> Loss = [[r + gamma * Qpolicy(s',a')] - Qnet(s,a)]^2... a' = argmax Q(s',a')
                if opt == 0:
                    with torch.no_grad():
                        Qnet_val = self.Qnet(s_next_batch)
                        _, ind = torch.max(Qnet_val, dim=1)   # we need just max of each row

                        Qpolicy_val = self.Qpolicy(s_next_batch)
                        Qpolicy_sa = torch.empty(self.batch_size)
                        for i, qval in enumerate(Qpolicy_val):
                            Qpolicy_sa[i] = qval[int(ind[i].item())]
                        Qpolicy_sa = Qpolicy_sa.to(device=self.device)
                        td_target = r_batch + self.gamma * Qpolicy_sa

                    # cur_Qnet_val --> Qnet(s,a)
                    s_batch = s_batch.to(device=self.device)
                    Qnet_val = self.Qnet(s_batch)
                    Qnet_sa = torch.empty(self.batch_size)
                    for i, qval in enumerate(Qnet_val):
                        Qnet_sa[i] = qval[int(a_batch[i].item())]

                    # td_error --> Loss = [td_target - Qnet(s,a)]^2
                    td_target = td_target.to(device=self.device)
                    Qnet_sa = Qnet_sa.to(device=self.device)
                    loss = self.criterion(Qnet_sa, td_target)
                   
                # td_error --> Loss = [[r + gamma * Qnet(s',a')] - Qpolicy(s,a)]^2... a' = argmax Qpolicy(s',a')
                else:
                    with torch.no_grad():
                        Qpolicy_val = self.Qpolicy(s_next_batch)
                        _, ind = torch.max(Qpolicy_val, dim=1)   # we need just max of each row
                        
                        Qnet_val = self.Qnet(s_next_batch)
                        Qnet_sa = torch.empty(self.batch_size)
                        for i, qval in enumerate(Qnet_val):
                            Qnet_sa[i] = qval[int(ind[i].item())]        
                        Qnet_sa = Qnet_sa.to(device=self.device)
                        td_target = r_batch + self.gamma * Qnet_sa

                    # cur_Qnet_val --> Qnet(s,a)
                    s_batch = s_batch.to(device=self.device)
                    Qpolicy_val = self.Qpolicy(s_batch)
                    Qpolicy_sa = torch.empty(self.batch_size)
                    for i, qval in enumerate(Qpolicy_val):
                        Qpolicy_sa[i] = qval[int(a_batch[i].item())]

                    # td_error --> Loss = [td_target - Qnet(s,a)]^2
                    td_target = td_target.to(device=self.device)
                    Qpolicy_sa = Qpolicy_sa.to(device=self.device)
                    loss = self.criterion(Qpolicy_sa, td_target)
                
                # update Qnet   ## Q[s][a] = Q[s][a] + alpha * (td_error).. alpha is weight gradients
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.Qnet.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
        
                # update state
                s = s_next 

                # plot on tensorboard 
                self.writer.add_scalar('Training_Steps vs Epsilon', self.epsilon, self.train_steps)

                # save progress to disk
                if self.train_steps % self.Qpolicy_update_interval == 0:
                    # self.Qpolicy.load_state_dict(self.Qnet.state_dict())  # in DDQN, no need to do this
                    self.save_current_model()
                    
            if self.episodes % 20000 == 0:
                self.save_replay_buffer()

            # save progress to disk and print
            if self.episodes % self.avg_over_episodes == 0:
                avg_reward = total_reward / self.avg_over_episodes
                total_reward = 0
                print (f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Loss: {loss.item():.4f}, Epsilon: {self.epsilon}')
                self.logger.info(f'Episode [{self.episodes}], Step [{self.train_steps}], Avg_Reward: [{avg_reward}], Loss: {loss.item():.4f}, Epsilon: {self.epsilon}')

                # plot on tensorboard
                # writer.add_scalar(label, y, x))
                self.writer.add_scalar('Episodes vs Epsilon', self.epsilon, self.episodes)
                self.writer.add_scalar('Episodes vs Avg_Reward', avg_reward, self.episodes)
                self.writer.add_scalar('Training_Steps vs Avg_Reward', avg_reward, self.train_steps)
                self.writer.add_scalar('Training_Steps vs Loss', loss, self.train_steps)
                self.writer.add_scalar('Training_Steps vs Episodes', self.episodes, self.train_steps)
                
            
        ###########################





"""
Report Ref:
1. To choose mini-batch size 
https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/

2.
"""

