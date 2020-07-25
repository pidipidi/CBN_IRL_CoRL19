# -*- coding: utf-8 -*-
import numpy as np
import pickle
import copy

class valueIterAgent:
    def __init__(self, n_actions=1, n_states=2, roadmap=None, skdtree=None,
                 states=None, poses=None, rewards=None, gamma=0.9, T=None, verbose=False,
                 **kwargs):
        """ """
        self.n_actions = n_actions #env.action_space.shape[0]
        self.n_states  = n_states #len(states)
        
        self.roadmap   = roadmap
        self.states    = states
        self.poses     = poses

        self.gamma     = gamma
        self.rewards   = rewards 
        self.policy    = None
        self.values    = None
        
        self.T         = copy.copy(T) #self.create_transition_matrix()

        self.skdtree   = skdtree
        self.theta     = None

        self.verbose   = verbose

    def create_transition_matrix(self):
        action_prob = 0.8
        T = np.ones((self.n_states, self.n_actions, self.n_actions))*(1.-action_prob)/float(self.n_actions-1)
        
        for s in xrange(self.n_states):
            for i in range(self.n_actions):
                T[s,i,i] = action_prob
        return T

    def solve_mdp_old(self, error=1e-10, init_values=None, env=None, reward_shaping=False,
                      return_params=False, **kwargs):
        
        if init_values is None:
            values = np.zeros(self.n_states)
        else:
            values = copy.copy(init_values)
            
        q_mat  = np.zeros((self.n_states, self.n_actions))

        # update each state
        diff = float("inf")
        count    = 0
        while diff > error:
            count += 1
            diff = 0.
            for s in xrange(self.n_states):
                ## max_v = float("-inf")
                # get idx of near states
                ids = self.roadmap[s]
                if reward_shaping is False:
                    q_mat[s] = np.dot(self.T[s], self.rewards[ids] + self.gamma*values[ids])
                else:
                    q_mat[s] = np.dot(self.T[s], self.rewards[ids] +
                                      env.get_distance_reward(self.states[s], self.states[ids]) +
                                      self.gamma*values[ids])

                max_v = np.amax(q_mat[s])
                                    
                new_diff = abs(values[s] - max_v)
                if new_diff > diff:                    
                    diff = new_diff
                values[s] = max_v
            if self.verbose: print diff
        if self.verbose: print "error converged to {} after {} iter".format(diff, count)

        if return_params:
            return values, {'q': q_mat}
        return values


    def solve_mdp(self, error=1e-10, init_values=None, env=None, reward_shaping=False,
                  return_params=False, **kwargs):
        
        if init_values is None:
            values = np.zeros(self.n_states)
        else:
            values = init_values #copy.copy(init_values)
        T       = copy.copy(kwargs.get('T', self.T))
        if 'T' in kwargs.keys() and 'goal' in kwargs.keys():
            s = kwargs['goal']
            T[s][0] = 0.
            T[s][0][0] = 1.
        
        r_mat   = self.rewards[self.roadmap]
        max_cnt = kwargs.get('max_cnt', None)

        # update each state
        diff  = float("inf")
        count = 0
        while True: #diff > error:
            count += 1
            
            # SxAxA, SxA + SxA
            r     = r_mat + self.gamma*values[self.roadmap]
            q_mat = np.multiply(T, r[:,np.newaxis,:]).sum(axis=-1)
            max_v = np.amax(q_mat, axis=-1)

            if max_cnt is None:
                new_diff = np.abs(values - max_v)
                diff = np.amax(new_diff)
                
                if diff <= error:
                    values = max_v
                    break
                
            elif count>max_cnt:
                values = max_v
                break
            
            values = max_v
            
            #if self.verbose: print diff
        if self.verbose: print "error converged to {} after {} iter".format(diff, count)
        
        if return_params: return values, {'q': q_mat}
        return values


    def find_policy(self, error=1e-10, env=None, reward_shaping=False):

        values = self.solve_mdp(error, env=env, reward_shaping=reward_shaping)                

        # generate stochastic policy
        policy = np.zeros([self.n_states, self.n_actions])            
        for s in xrange(self.n_states):
            ids = self.roadmap[s]
            for a in xrange(self.n_actions):
                if reward_shaping:
                    dist_r = env.get_distance_reward(self.states[s], self.states[ids[a]])
                else:
                    dist_r = 0

                policy[s,a] = np.dot(self.T[s, a], self.rewards[ids]+dist_r+
                                     self.gamma*values[ids])

        #policy      -= policy.max(axis=1).reshape((self.n_states, 1))  # For numerical stability.
        policy = np.exp(policy)/np.exp(policy).sum(axis=1).reshape((self.n_states, 1))
        self.policy = policy
        self.values = values
        return self.policy, self.values

    def get_policy(self):
        return self.policy

    def get_rewards(self):
        return self.rewards

    def set_rewards(self, rewards):
        self.rewards = np.array(rewards)

    def set_goal(self, s):
        self.T[s][0] = 0.
        self.T[s][0][0] = 1.

    def act(self, state, s_idx=None, e=0., return_next_state=False):
        if type(state) is list: state = np.array(state)
        if s_idx is None:
            s_idx = np.argmin(np.linalg.norm(self.states-state, axis=-1))
        values      = self.policy[s_idx]
        best_action = np.argmax(values)
        
        # get nearstates
        s_idx     = self.roadmap[s_idx][best_action]        
        nxt_state = np.array(self.states[s_idx])

        if return_next_state:
            return nxt_state
        return nxt_state-state

    def reset(self):
        return

    def save(self, filename):
        d = {'policy': self.policy, 'rewards': self.rewards, 'roadmap': self.roadmap,\
             'T': self.T, "n_actions": self.n_actions, "n_states": self.n_states,
             'states': self.states, 'values': self.values, 'theta': self.theta,\
             'poses': self.poses}
        pickle.dump( d, open( filename, "wb" ) )
        pickle.dump( self.skdtree, open( filename+'_kdtree', "wb" ) )
        print "Saved a file: ", filename

    def load(self, filename):
        d = pickle.load( open( filename, "rb" ) )
        self.policy  = d['policy']
        self.values  = d['values']
        self.rewards = d['rewards']
        self.roadmap = d['roadmap']
        self.T       = d['T']
        self.n_actions = d['n_actions']
        self.n_states  = d['n_states']
        self.states  = d['states']
        self.poses  = d['poses']
        self.theta   = d['theta']
        self.skdtree = pickle.load( open( filename+'_kdtree', "rb") )
        self.reset()

        if type(self.states) is not list: self.states = self.states.tolist()
        print "Loaded a file: ", filename



