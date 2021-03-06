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

def make_default_reward(goal, dist_thres=2.5):
    def reward_fn(state):
        dist = np.linalg.norm(goal-state)
        if dist <= dist_thres: r = 1
        else: r=0.        
        return r
    return reward_fn
    


class ReachingEnv_v1(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 20
        }
    def __init__(self, start=None, goal=None, objects=None,
                 **kwargs):
        self.robot_size = 0.5
        self.goal_size  = 1.
        self.force_mag = 0.5

        self.set_start_state(start)
        self.set_goal_state(goal)
        self.set_objects(objects)
        self.state = start
        self.reward_fn = make_default_reward(goal)
        ## self.rewards   = None
        self.trees = None
        
        self.roadmap = kwargs.get('roadmap', None)        
        self.states  = kwargs.get('states', None)
        self.steps_beyond_done = None

        self.action_space = spaces.Discrete(4)
        self.fig = None

        self.seed(0)

    @property
    def observation_space(self):
        return spaces.Box( 0., 60., (2,), dtype=np.float16)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):
        ''' return next observation, reward, finished, success '''
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        ## state  = self.state + action
        
        # 0: +x +y
        # 1: +x -y
        # 2: -x +y
        # 3: -x -y
        if action==0:
            force = np.array([self.force_mag, self.force_mag])
        elif action==1:
            force = np.array([self.force_mag, -self.force_mag])
        elif action==2:
            force = np.array([-self.force_mag, self.force_mag])
        else:
            force = np.array([-self.force_mag, -self.force_mag])
        state = self.state + force

        info = {}
        if self.isValid(state, check_collision=True):
            # succeeded state        
            ## info['success'] = True
            # terminal state        
            done = self.isGoal(self.state)
        else:
            # failed state        
            ## info['success'] = False
            state = self.state
            ## state = np.clip(state, self.observation_space.low, self.observation_space.high)
            done = True

        if not done:
            reward = self.get_reward(state)            
        elif self.steps_beyond_done is None:
            # collision or out of bound?
            self.steps_beyond_done = 0
            reward = self.get_reward(state)            
        elif done and self.steps_beyond_done == 0:
            logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
            
        self.state = state
        return self.state, reward, done, info
            

    def reset(self):
        ## from IPython import embed; embed(); sys.exit()
        self.state = self.start_state + self.np_random.uniform(low=-1., high=1.,
                                                               size=self.observation_space.shape)
        self.steps_beyond_done = None        
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
        plt.close(fig=self.fig)
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
        ## return self.rewards[state]
        return self.reward_fn(state)
    def get_rewards(self, states):
        return self.reward_fn(states)
        ## r = - np.linalg.norm(states-self.goal_state, axis=1)
        ## return r
    def get_distance_reward(self, s, s_next, eta=15.0):
        if len(np.shape(s_next))==2:
            r1 = -eta * np.linalg.norm(self.goal_state-s_next, axis=1)
        else:
            r1 = -eta * np.linalg.norm(self.goal_state-s_next)
        r2 = -eta * np.linalg.norm(self.goal_state-s)
        return r1-r2        
    def get_pos(self, state):
        return state
    def get_states(self):
        return self.states    
    def get_label_trees(self):
        return self.trees, self.objects_in_tree

    def get_progress(self, state):

        if type(state) is list: state = np.array(state)
        if self.discrete_progress:
            p = 1. - np.linalg.norm(self.goal_state-state, axis=-1)/\
              (np.linalg.norm(self.start_state-state, axis=-1) + \
               np.linalg.norm(self.goal_state-state, axis=-1) )                        
            if type(p) is np.ndarray:
                progress = p
                dists_list = []
                for key in self.trees.keys():
                    _,dists = self.trees[key].search(state.T,1)
                    dists_list.append(dists)
                dists = np.amin(dists_list, axis=0)

                ## ids   = np.argsort(progress)
                ## dists = dists[ids]
                ## p     = p[ids]
                                
                for i in range(len(p)):                    
                    if p[i]<1.5 and dists[i]>5: progress[i] = 0.
                    ## elif p[i]>=0.5 and dists[i]>5: progress[i] = 1.
                    else: progress[i] = 1.
            else:
                d = np.amin(np.linalg.norm(self.objects-state, axis=-1))
                if p<1.5 and d>5: progress = 0.
                ## elif p>=0.5 and d>5: progress = 1.
                else: progress = 1.
        else:
            progress = 1. - np.linalg.norm(self.goal_state-state, axis=-1)/(np.linalg.norm(self.start_state-state, axis=-1) + np.linalg.norm(self.goal_state-state, axis=-1) )
            #progress = 1. - np.linalg.norm(self.goal_state-state, axis=-1)/np.linalg.norm(self.start_state-self.goal_state)
            #progress = np.linalg.norm(self.goal_state-state)
            ## progress = self.observation_space.high[0]*np.sqrt(2)-np.linalg.norm(self.goal_state-state)
            ## progress /= (self.observation_space.high[0]*np.sqrt(2))

            #progress = np.clip(progress, a_min=0., a_max=1.)

        return progress


    def set_label_trees(self, trees, objects):
        self.trees = trees
        self.objects_in_tree = objects
        
    def set_start_state(self, state):
        self.start_state = np.array(copy.deepcopy(state))
    def set_goal_state(self, state):
        self.goal_state = np.array(copy.deepcopy(state))
    def set_objects(self, objs):
        self.objects = np.array(objs)
    ## def set_rewards(self, rewards):
    ##     self.rewards = rewards
    def set_reward(self, reward_fn):
        self.reward_fn = reward_fn
    def set_states(self, states):
        self.states = states
