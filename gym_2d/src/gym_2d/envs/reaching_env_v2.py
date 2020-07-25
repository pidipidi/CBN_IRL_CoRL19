import gym
from gym import error, spaces, utils
from gym.utils import seeding

import time, copy
import random
import math
import numpy as np
import scipy.spatial

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_default_reward(goal, dist_thres=2.5, limits=[0,1]):
    def reward_fn(state):
        ## if state[0]<=limits[0] or state[0] >= limits[1]: return -1
        ## if state[1]<=limits[0] or state[1] >= limits[1]: return -1
        dist = np.linalg.norm(goal-state)
        if dist <= dist_thres: r = 100.
        else: r=0.        
        return r
    return reward_fn
    
## def make_default_reward(goal, dist_thres=2.5, limits=[0,1]):
##     def reward_fn(state):
##         ## if state[0]<=limits[0] or state[0] >= limits[1]: return -1
##         ## if state[1]<=limits[0] or state[1] >= limits[1]: return -1
##         dist = np.linalg.norm(goal-state)
##         return 1./dist/dist*dist_thres
##         #return np.exp(-0.1*dist)*100.
##     return reward_fn



class ReachingEnv_v2(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 20
        }
    def __init__(self, start=None, goal=None, objects=None, **kwargs):
        self._max_episode_steps=200

        self.robot_size = 0.025
        self.goal_size  = 0.025
        self.start_state = None
        self.goal_State  = None
        self.data_dict = kwargs.get('data_dict', None)

        self.set_start_state(start)
        self.set_goal_state(goal)
        self.set_objects(objects)
        self.reward_fn = make_default_reward(goal, dist_thres=self.goal_size)
        self.state = start
        ## self.rewards   = None
        self.trees = None
        self.reset_method= "random" #'start'
        
        self.roadmap = kwargs.get('roadmap', None)        
        self.states  = kwargs.get('states', None)

        self.fig = None

        self.seed(0)
        #self.reset()


    @property
    def action_space(self):
        # this will be the joint space
        return spaces.Box( -1., 1., (2,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box( 0., 1., (2,), dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        ''' return next observation, reward, finished, success '''
        ## assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        ## print self.state, action
        
        state  = self.state + np.clip(action, self.action_space.low, self.action_space.high)/25.
        state  = np.clip(state, self.observation_space.low, self.observation_space.high)

        validity = self.isValid(state, check_collision=False)
        if validity:
            # succeeded state        
            ## info['success'] = True
            # terminal state        
            done = self.isGoal(state)
        else:
            # failed state        
            ## info['success'] = False
            state = self.state
            ## state = np.clip(state, self.observation_space.low, self.observation_space.high)
            done = True

        reward = self.get_reward(state)
        #reward = -( np.linalg.norm(self.goal_state-state)**2 + 0.0001*np.sum(action**2) )
        if validity is False:
            reward -= 1.
        ## if state[0]<=self.observation_space.low[0] or state[0] >= self.observation_space.high[0] or\
        ##   state[1]<=self.observation_space.low[1] or state[1] >= self.observation_space.high[1]:            
        ##     reward -= 1.
        
        self.state = state
        return self.state, reward, done, {}
            

    def reset(self):
        """ """
        if self.reset_method=='start':
            self.state = self.start_state + self.np_random.uniform(low=-0.2,
                                                                   high=0.2,
                                                                   size=self.observation_space.shape)
        elif self.reset_method=='goal':
            self.state = self.goal_state + self.np_random.uniform(low=-0.2,
                                                                  high=0.2,
                                                                  size=self.observation_space.shape)
        elif self.reset_method=='linear':
            ## from IPython import embed; embed(); sys.exit()
            offset = np.random.ranf()*(self.goal_state-self.start_state)
            self.state = self.start_state + offset+self.np_random.uniform(low=-0.2,
                                                                          high=0.2,
                                                                          size=self.observation_space.shape)
        else:
            self.state = self.np_random.uniform(low=self.observation_space.low[0],
                                                high=self.observation_space.high[1],
                                                size=self.observation_space.shape)
        self.state  = np.clip(self.state, self.observation_space.low, self.observation_space.high)
            
        return self.state

    
    def isGoal(self, state):
        """Check goal"""
        if np.linalg.norm(state-self.goal_state) < self.goal_size:
            return True
        return False
    
    
    def isValid(self, state, check_collision=False):
        """Check validity of current state"""
        # work space
        if any(state[i] < self.observation_space.low[i] for i in range(self.observation_space.shape[0])):
            return False
        if any(state[i] > self.observation_space.high[i] for i in range(self.observation_space.shape[0])):
            return False

        # collision # TODO: use tree??
        if self.trees is not None and check_collision:
            for key in self.trees.keys():
                _, dist = self.trees[key].search(state)
                if dist < self.robot_size:
                    return False
        return True

    def render(self, mode="human"):
        ''' Render 2d map
        '''
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')

            #colors = cm.rainbow(np.linspace(0, 1, len(self.objects_in_tree.keys())))
            ## for i, key in enumerate(self.objects_in_tree.keys()):
            ##     plt.plot(self.objects_in_tree[key][:,0], self.objects_in_tree[key][:,1], 'o', c=colors[i])
            plt.plot(self.objects[:,0], self.objects[:,1], 'ko')
            
            plt.plot(self.start_state[0], self.start_state[1], 'rx')
            plt.plot(self.goal_state[0], self.goal_state[1], 'r^')        
            
            plt.xlim(self.observation_space.low[0], self.observation_space.high[0] )
            plt.ylim(self.observation_space.low[1], self.observation_space.high[1] )
            plt.show(block=False)


        if self.state is None: return None
        
        # plot current point?
        p, = plt.plot(self.state[0], self.state[1], '.b')
        plt.pause(0.00001)
        ## self.fig.canvas.flush_events()
        p.remove()
        return


    def close(self):
        ## plt.close(fig=self.fig)
        plt.close()
        self.fig = None 


    # --------------------- Get/Set -----------------------
    ## def get_action(self, state1, state2):
    ##     return state2-state1
    def get_start_state(self):
        return self.start_state
    def get_goal_state(self):
        return self.goal_state
    def get_objects(self):
        return self.objects
    def get_objects_in_tree(self):
        return self.objects_in_tree
    def get_reward(self, state):
        return self.reward_fn(state)
    def get_rewards(self, states):
        return self.reward_fn(states)
    def get_reward_fn(self):
        return self.reward_fn
    def get_distance_reward(self, s, s_next, eta=15.0):
        if len(np.shape(s_next))==2:
            r1 = -eta * np.linalg.norm(self.goal_state-s_next, axis=1)
        else:
            r1 = -eta * np.linalg.norm(self.goal_state-s_next)
        r2 = -eta * np.linalg.norm(self.goal_state-s)
        return r1-r2        
    def get_label_trees(self):
        return self.trees, self.objects_in_tree

    def set_label_trees(self, trees, objects):
        self.trees = trees
        self.objects_in_tree = objects
    def set_start_state(self, state):
        self.start_state = np.array(copy.deepcopy(state))
    def set_goal_state(self, state):
        self.goal_state = np.array(copy.deepcopy(state))
    def set_objects(self, objs):
        self.objects = np.array(objs)
    def set_reward_fn(self, reward_fn):
        self.reward_fn = reward_fn
