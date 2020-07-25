# -*- coding: utf-8 -*-
import numpy as np
import pickle
import sharedmem
## import multiprocessing as mp
from cbn_irl.mdp import value_iteration as vi
## from mdp import constrained_value_iteration as cvi
import copy

eps = np.finfo(np.float).eps
## eps = np.finfo(float).eps    

def precomputeQ(mdp, support_states, error=1e-10, support_features=None, support_feature_state_dict=None,
                cstr_fn=None, add_no_cstr=True, **kwargs):
    """
    Compute Q using single process
    """
    n_support_states = len(support_states)
    n_states, n_actions = mdp.n_states, mdp.n_actions

    support_q_mat = {}
    support_values = {}
    support_validity = {}

    if support_features is not None:
        support_feature_ids, support_feature_values = support_features
        computed_f_id = []

    # Get constraint-based transition matrix
    if cstr_fn is not None and len(cstr_fn)>0:

        feat_map = kwargs['feat_map']
        roadmap  = kwargs['roadmap']
        states   = mdp.states

        n_cstr_fn = len(cstr_fn)
        cstr_T = []
        for i in range(n_cstr_fn):
            T            = copy.copy(mdp.T)
            validity_map = cstr_fn[i](None, f=feat_map)
            T *= validity_map[roadmap][:,np.newaxis,:]
            sum_T = np.sum(T, axis=-1)
            sum_T[np.where(sum_T==0.)] = 1.
            T /= (sum_T[:,:,np.newaxis]+eps)
            #from IPython import embed; embed(); sys.exit()  
            cstr_T.append(T)
    else:
        n_cstr_fn = 0
                    
    org_T = copy.copy(mdp.T)
    max_q = -np.inf
    for i, s in enumerate(support_states):
        if i%10==0: print i
        ## if i < len(support_states)-1: continue
            
        # set reward goal=s
        rewards = mdp.get_rewards()
        if rewards is None: rewards=np.zeros(len(mdp.states))
        rewards = np.array([0. if r>0. else r for r in rewards ])

        if support_features is None:
            if rewards[s] >= 0.:
                rewards[s] = 1. 
        else:
            # find all states that gives f_g in states
            f_id           = support_feature_ids[i]
            goal_state_ids = support_feature_state_dict[f_id]
            if f_id in computed_f_id:
                support_q_mat[s]  = support_q_mat[min(goal_state_ids)]
                support_values[s] = support_values[min(goal_state_ids)]
                continue
            else:
                computed_f_id.append(f_id)
                
            for idx in goal_state_ids:
                if rewards[idx] >= 0.:
                    rewards[idx] = 1. 

        mdp.set_rewards(rewards)

        if cstr_fn is not None:
            support_q_mat[s] = []
            support_values[s]= []
            support_validity[s]= []
            for j in range(n_cstr_fn):
                ## from IPython import embed; embed(); sys.exit()                
                # check if the goal is isolated
                if np.sum(cstr_T[j][goal_state_ids,0,0])==len(goal_state_ids):
                    support_q_mat[s].append( np.zeros((n_states, n_actions)) )
                    support_values[s].append( np.zeros(n_states) )
                else:
                    # solve mdp
                    values, param_dict = mdp.solve_mdp(error, T=cstr_T[j], goal=s, return_params=True)
                    support_q_mat[s].append( copy.copy(param_dict['q']) )
                    support_values[s].append( copy.copy(values) )
                    ## support_validity[s].append( cstr_fn[j](mdp.states[s]) )
                    support_validity[s].append( cstr_fn[j](s) )

            if add_no_cstr:
                # no constraint
                values, param_dict = mdp.solve_mdp(error, return_params=True)
                support_q_mat[s].append( copy.copy(param_dict['q']) )
                support_values[s].append( copy.copy(values) )
                support_validity[s].append( True )

        else:
            mdp.T = copy.copy(org_T)
            mdp.set_goal(s)
            
            values, param_dict = mdp.solve_mdp(error, return_params=True)
            support_q_mat[s]  = copy.copy(param_dict['q']) 
            support_values[s] = copy.copy(values) 

        ## if i==len(support_states)-1:
        ##     sys.path.insert(0,'../..')
        ##     from viz import viz as v
        ##     v.reward_value_plot(rewards, values, mdp.states)
        ##     from IPython import embed; embed(); sys.exit()
        ##     sys.exit()

    # scaling
    ## for key in support_q_mat.keys():
    ##     for i in range(len(support_q_mat[key])):
    ##         support_q_mat[key][i] /= max_q

    if cstr_fn is not None:
        return support_q_mat, support_values, support_validity
    else:
        return support_q_mat, support_values




def computeQ(mdp, support_states, error=1e-10, support_features=None, support_feature_state_dict=None,
             cstr_fn=None, add_no_cstr=True, max_cnt=100, **kwargs):
    """Compute Q using multi-process
    """
    # initialization of variables
    n_support_states = len(support_states)
    n_actions, n_states = mdp.n_actions, mdp.n_states

    roadmap = mdp.roadmap
    states  = mdp.states
    gamma   = mdp.gamma
    T       = mdp.T
    rewards = mdp.get_rewards()
    #from IPython import embed; embed(); sys.exit()    
    if rewards is None: rewards=np.zeros(len(mdp.states))
    else:
        rewards = np.array(rewards)
        rewards[np.where(rewards>0)]=0.        
    support_state_ids = np.arange(n_support_states, dtype='i')
    
    if support_features is not None:
        support_feature_ids, support_feature_values = support_features        
        computed_f_id = sharedmem.full(len(support_feature_values), False, dtype='b')
    else:
        return NotImplementedError

    if cstr_fn is None:
        support_q_mat    = sharedmem.full((n_support_states, mdp.n_states, mdp.n_actions), 0.)
        support_values   = sharedmem.full((n_support_states, mdp.n_states), 0.)
        support_validity = sharedmem.full((n_support_states), True)
    else:
        if add_no_cstr: n_cstr_fn = len(cstr_fn) + 1
        else:           n_cstr_fn = len(cstr_fn) 
        support_q_mat    = sharedmem.full((n_support_states, n_cstr_fn, mdp.n_states,
                                           mdp.n_actions), 0.)
        support_values   = sharedmem.full((n_support_states, n_cstr_fn, mdp.n_states), 0.)
        support_validity = sharedmem.full((n_support_states, n_cstr_fn), True)

        if len(cstr_fn)>0:
            feat_map = kwargs['feat_map']
            roadmap  = kwargs['roadmap']
            states   = mdp.states

            cstr_T = []
            for i in range(len(cstr_fn)):
                validity_map      = cstr_fn[i](None, f=feat_map)[roadmap]
                validity_map[:,0] = True
                Tc                = mdp.T*validity_map[:,np.newaxis,:]
                Tc[:,:,0]         = eps
                sum_T             = np.sum(Tc, axis=-1)
                Tc               /= (sum_T[:,:,np.newaxis]+eps)
                cstr_T.append(Tc)
        
    # Start multi processing over support states
    with sharedmem.MapReduce() as pool:
        if n_support_states % sharedmem.cpu_count() == 0:
            chunksize = n_support_states / sharedmem.cpu_count()
        else:
            chunksize = n_support_states / sharedmem.cpu_count() + 1

        def work(i):
            state_ids = support_state_ids[slice (i, i + chunksize)]
                
            new_rewards = copy.copy(rewards)
            values      = np.zeros(n_states)
            
            for j, state_id in enumerate(state_ids):
                s = support_states[state_id] # state id in states

                # vi agent
                mdp = vi.valueIterAgent(n_actions, n_states,
                                        roadmap, None, states,
                                        gamma=gamma, T=T)                
                mdp.set_goal(s)

                if support_feature_ids is None:
                    if new_rewards[s] >= 0.:
                        new_rewards[s] = 1. 
                else:
                    # find all states that gives f_g in states
                    f_id              = support_feature_ids[state_id]
                    goal_state_ids    = support_feature_state_dict[f_id]
                    
                    if computed_f_id[f_id]:
                        continue
                    else:
                        computed_f_id[f_id] = True
                    new_rewards[goal_state_ids] = 1.
                mdp.set_rewards(new_rewards)

                # Store q_mat and validity mat per state
                if cstr_fn is not None:
                    for k in range(len(cstr_fn)):
                        # check if the goal is isolated
                        if np.sum(cstr_T[k][goal_state_ids])>0.:
                            values, param_dict = mdp.solve_mdp(error, init_values=values,
                                                               T=cstr_T[k], max_cnt=max_cnt,
                                                               goal=s,
                                                               return_params=True)
                            support_q_mat[state_id][k]    = param_dict['q']
                            support_validity[state_id][k] = cstr_fn[k](s)
                            support_values[state_id][k]   = values

                    if add_no_cstr:
                        values, param_dict = mdp.solve_mdp(error, init_values=values,
                                                           T=T, max_cnt=max_cnt,
                                                           ## goal=s,
                                                           return_params=True)
                        support_q_mat[state_id][-1]    = param_dict['q']
                        support_validity[state_id][-1] = True 
                        support_values[state_id][-1]   = values
                else:
                    values, param_dict = mdp.solve_mdp(error, init_values=values,
                                                       max_cnt=max_cnt,
                                                       return_params=True)
                    support_q_mat[state_id]  = param_dict['q']                    
                    support_values[state_id] = values
                    
                # find all states that gives f_g in states
                for gs in goal_state_ids:
                    k = support_states.index(gs)
                    if k!=state_id:
                        support_q_mat[k] = support_q_mat[state_id]
                # reset
                ## new_rewards = copy.copy(rewards)
                if support_feature_ids is None:
                    new_rewards[s] = 0. 
                else:
                    new_rewards[goal_state_ids] = 0.

        pool.map(work, range(0, n_support_states, chunksize))#, reduce=reduce)

    # convert sharedmem array to dict
    support_q_mat_dict    = {}
    support_values_dict   = {}
    support_validity_dict = {}
    for i, s in enumerate(support_states):
        support_q_mat_dict[s]  = np.array(support_q_mat[i]) #.astype(np.float)
        support_values_dict[s]  = np.array(support_values[i]) #.astype(np.float)

        if cstr_fn is not None:
            support_validity_dict[s] = np.array(support_validity[i])

    if cstr_fn is not None:
        return support_q_mat_dict, support_values_dict, support_validity_dict
    else:
        return support_q_mat_dict, support_values_dict



def computePolicies(mdp, goal_states, error=1e-10):
    """Compute Q using multi-process
    """
    # initialization of variables
    n_goal_states = len(goal_states)
    n_actions, n_states = mdp.n_actions, mdp.n_states

    roadmap = mdp.roadmap
    states  = mdp.states
    gamma   = mdp.gamma
    T       = mdp.T
    rewards = mdp.get_rewards()
    #from IPython import embed; embed(); sys.exit()    
    if rewards is None: rewards=np.zeros(len(mdp.states))
    else:
        rewards = np.array(rewards)
        rewards[np.where(rewards>0)]=0.        
    goal_state_ids = np.arange(n_goal_states, dtype='i')
    
    goal_policy_mat  = sharedmem.full((n_goal_states, mdp.n_states, mdp.n_actions), 0.)
    ## goal_values   = sharedmem.full((n_goal_states, mdp.n_states), 0.)
    ## goal_validity = sharedmem.full((n_goal_states), True)
        
    # Start multi processing over goal states
    with sharedmem.MapReduce() as pool:
        if n_goal_states % sharedmem.cpu_count() == 0:
            chunksize = n_goal_states / sharedmem.cpu_count()
        else:
            chunksize = n_goal_states / sharedmem.cpu_count() + 1

        def work(i):
            state_ids = goal_state_ids[slice (i, i + chunksize)] #0,1,2,3
                
            new_rewards = copy.copy(rewards)
            values      = np.zeros(n_states)
            
            for j, goal_state_id in enumerate(state_ids):
                s = goal_states[goal_state_id] # state id in states
                ## s_idx = states.index(s)
                
                # vi agent
                mdp = vi.valueIterAgent(n_actions, n_states,
                                        roadmap, None, states,
                                        gamma=gamma, T=T)                
                mdp.set_goal(s)

                if new_rewards[s] >= 0.:
                    new_rewards[s] = 1. 
                mdp.set_rewards(new_rewards)

                # Store q_mat and validity mat per state
                policy, _ = mdp.find_policy(error)
                goal_policy_mat[goal_state_id] = policy                    
                    
                # find all states that gives f_g in states
                for k, gs in enumerate(goal_states):
                    if s == gs:
                        goal_policy_mat[k] = policy
                # reset
                new_rewards[s] = 0. 

        pool.map(work, range(0, n_goal_states, chunksize))#, reduce=reduce)

    # convert sharedmem array to list
    policies = []    
    for i in range(n_goal_states):
        policies.append( np.array(goal_policy_mat[i]) )

    return policies



## def solve_chunk(segmented_state_ids, support_states, roadmap, states, rewards, gamma, T, error,
##                 n_states, n_actions,
##                 support_feature_ids=None, support_feature_values=None, computed_f_id=None):

##     # vi agent
##     mdp = vi.valueIterAgent(n_actions, n_states,
##                             roadmap, None, states,
##                             gamma=gamma, T=T)

##     support_q_mat = {}
##     #support_values = {}
##     #support_validity = {}

##     for i in segmented_state_ids:
##         s = support_states[i]
##         rewards = np.array([0. if r>0. else r for r in rewards ])
    
##         if support_feature_ids is None:
##             if rewards[s] >= 0.:
##                 rewards[s] = 1. 
##         else:
##             # find all states that gives f_g in states
##             f_id              = support_feature_ids[i]
##             goal_state_ids    = support_feature_state_dict[f_id]
##             ## goal_state_ids    = get_state_goal_from_feature(f_id, support_states, support_feature_ids)
##             if f_id in computed_f_id:
##                 for gs in goal_state_ids:
##                     if gs in support_q_mat.keys():
##                         support_q_mat[s]  = support_q_mat[gs]
##                         support_values[s] = support_values[gs]
##                 continue
##             else:
##                 computed_f_id.append(f_id)
                
##             for idx in goal_state_ids:
##                 if rewards[idx] >= 0.:
##                     rewards[idx] = 1. 

##         mdp.set_rewards(rewards)

##         if cstr_fn is not None:
##             return NotImplementedError
##             ## support_q_mat[s] = []
##             ## support_values[s]= []
##             ## support_validity[s]= []
##             ## for j in range(len(cstr_fn)):
##             ##     # solve mdp
##             ##     values, q_mat = mdp.solve_mdp(error, cstr_fn=cstr_fn[j], return_q_mat=True, return_T=False)
##             ##     support_q_mat[s].append( copy.copy(q_mat) )
##             ##     support_values[s].append( copy.copy(values) )
##             ##     support_validity[s].append( cstr_fn[j](mdp.states[s]) )

##             ## if add_no_cstr:
##             ##     # no constraint
##             ##     values, q_mat = mdp.solve_mdp(error, return_q_mat=True, return_T=False)
##             ##     support_q_mat[s].append( copy.copy(q_mat) )
##             ##     support_values[s].append( copy.copy(values) )
##             ##     support_validity[s].append( True )
##         else:
##             values, q_mat = mdp.solve_mdp(error, return_q_mat=True)
##             support_q_mat[s] = copy.copy(q_mat) 
##             #support_values[s] = copy.copy(values) 

##     queue.put(support_q_mat)


## def segment_state_indices(num_states, num_processes):
##     # segments the state indices into chunks to distribute between processes
##     state_idxs = np.arange(num_states)
##     num_uneven_states = num_states % num_processes
    
##     if num_uneven_states == 0:
##         segmented_state_idxs = state_idxs.reshape(num_processes, -1)
##     else:
##         segmented_state_idxs = state_idxs[:num_states - num_uneven_states].reshape(num_processes, -1).tolist()
##         segmented_state_idxs[-1] = np.hstack((segmented_state_idxs[-1], state_idxs[-num_uneven_states:])).tolist()
        
##     return segmented_state_idxs

