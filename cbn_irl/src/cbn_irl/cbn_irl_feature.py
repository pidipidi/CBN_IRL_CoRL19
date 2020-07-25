import sys, os, copy, time
import random
import pickle
import numpy as np, scipy
import PyKDL
from tqdm import tqdm

#from mdp import parallel_value_iteration as vi
from mdp import value_iteration as vi
#from mdp import constrained_value_iteration as vi
from mdp import feature_utils as fu
import bn_irl_utils as ut
import bn_irl_common as bic
import bn_irl_q_fn as biq

from path_planning import dijkstra_planning 


def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def make_feature_fn(env, feature_fn):
    def fn(state):
        return feature_fn(env, state)
    return fn

def bn_irl(env, roadmap, skdtree, states, T, gamma, trajs, idx_trajs,
           feature_fn, alphas=(0.1,1.0), eta=0.5, punishment=0., Ts=[0.1, 0.7],
           num_feat=100, cstr_ths=2.33, window_size=5, **kwargs):
    """
    Bayesian Nonparametric Inverse Reinforcement Learning (BN IRL)

    inputs:
        gamma       float - RL discount factor
        trajs       a list of demonstrations
        lr          float - learning rate
        n_iters     int - number of optimization steps
        alphas      confidence for resampling and assignments
        eta         concentration (alpha in CRP)

    returns
        rewards     Nx1 vector - recoverred state rewards
    """
    n_goals = kwargs.get('n_goals', 3)
    burn_in = kwargs.get('burn_in', 1000)
    n_iters = kwargs.get('n_iter', 2000)
    n_iters = n_iters if n_iters > burn_in else int(n_iters*1.5)
    use_clusters     = True
    use_action_prior = kwargs.get('use_action_prior', False)
    alpha1, alpha2   = alphas
    max_cnt          = kwargs.get('max_cnt', 100)
    return_best      = False
    
    N_STATES         = len(states)
    N_ACTIONS        = len(roadmap[0])

    # support observations (i.e., demonstrations)
    observations     = ut.trajs2observations(trajs, idx_trajs, roadmap, states)
    support_states   = bic.get_support_space(observations)
    n_observations   = len(observations)
    n_support_states = len(support_states)

    observation_states  = [obs.state for obs in observations]
    observation_actions = [obs.action for obs in observations]

    # visualize feature distribution
    ## feat_map = fu.get_features_from_states(env, states, feature_fn)
    ## feat_trajs = np.array(ut.trajs2featTrajs(idx_trajs, feat_map))
    ## viz_feat_traj(feat_trajs[0])
    
    # precomputation Q and pi per support features that is a list of goals [feature, policy]
    time0 = time.time()
    if os.path.isfile(kwargs['sav_filenames']['Q']) is False or kwargs.get('vi_renew', True):
        assert len(np.shape(trajs))==3, "wrong shape or number of trajectories"

        # trajs, idx_trajs, feat_trajs have the same order
        feat_map = fu.get_features_from_states(env, states, feature_fn)
        feat_trajs = np.array(ut.trajs2featTrajs(idx_trajs, feat_map))

        # get features given each feature sub goal
        # support_feature: each support state's feature index, {indices | s_idx \in S, f(s)==f(s_g)}
        # support_feature_values: {feature values | ... }
        support_feature_ids, support_feature_values = bic.get_support_feature_space(support_states,
                                                                                    states, feat_map,
                                                                                    env)
        support_feature_state_dict = bic.get_feature_to_state_dict(support_states, support_feature_ids)

        # Select cstr_feat_id
        cstr_feat_id = []
        for i in range(np.shape(feat_trajs)[-1]):
            fs = feat_trajs[0,:,i]

            stds = []
            for j in range(len(fs)-window_size-1):
                stds.append( np.std(fs[j:j+window_size]) )
            stds = np.array(stds)
            print np.mean(stds*2.) , cstr_ths, cstr_ths/float(window_size)
            # check the average of variances over moving windows
            #if np.mean(stds*2.) < cstr_ths: #/float(window_size):
            if np.mean(stds*2) < cstr_ths: 
                cstr_feat_id.append(i)
        print "CSTR ID: ", cstr_feat_id

        #TODO: currently, we simplified feature range and pairs!        
        # discretize features
        f_min = np.amin(feat_trajs[0,:,cstr_feat_id], axis=-1)
        f_max = np.amax(feat_trajs[0,:,cstr_feat_id], axis=-1)
        feat_range = np.linspace(f_min, f_max, num_feat) #Nx2
        
        support_policy_dict   = {}
        support_validity_dict = {}
        
        # compute policies for states x features
        for i in range(len(feat_range)):
            print "f_idx = {}/{}".format(i, len(feat_range))
            # compute constraint function over observations
            feat = np.zeros(len(feat_map[0]))
            feat[cstr_feat_id] = feat_range[i]
            cstr_fns, cstr_params = make_train_cstr_fns(feat_map, feat, states,
                                                        cstr_feat_id, ths=cstr_ths)
            cstr_map   = cstr_fns[0](None, f=feat_map)*1.

            # if the score is 0, skip since these constraints do not make sense
            # 0th element is just needed for other processes 
            if sum(cstr_map[idx_trajs[0]]) == 0 and i>0 and False:
                support_policy_dict[i]   = {}
                support_validity_dict[i] = {}
                support_values = None
            else:
                # vi agent
                agent = vi.valueIterAgent(N_ACTIONS, N_STATES,
                                          roadmap, skdtree, states,
                                          rewards=kwargs.get('rewards', None),
                                          gamma=gamma, T=T)
                
                support_policy, support_values, support_validity =\
                  biq.computeQ(agent, support_states,
                               support_features=(support_feature_ids,
                                                 support_feature_values),
                                                 support_feature_state_dict=support_feature_state_dict,
                                                 cstr_fn=cstr_fns, add_no_cstr=False, max_cnt=max_cnt,
                                                 feat_map=feat_map, roadmap=roadmap)
                support_policy_dict[i]   = support_policy
                support_validity_dict[i] = support_validity

            #from IPython import embed; embed(); sys.exit()
            # Just to print out the constraint scores
            cstr_score = 0
            for idx in idx_trajs[0]:
                if support_values is None: continue
                cstr_score += cstr_map[idx]*support_values[idx_trajs[0][-1]][0,idx]
            print "{}th feat's cstr score = {}".format(i, cstr_score)

            from viz import viz as v
            ## ss = states[cstr_map>0]
            ## cc = cstr_map[cstr_map>0]
            ## if np.amax(support_values[idx_trajs[0][-1]]) == 0: continue
            v.reward_value_plot(agent.rewards, support_values[idx_trajs[0][-1]][0], states, trajs=trajs)
            ## v.reward_value_3d(cc, cc, ss, trajs=trajs, env=env)
            ## v.reward_value_plot(cstr_map, cstr_map, states, trajs=trajs)
            ## v.reward_value_3d(cstr_map, cstr_map, states, trajs=trajs, env=env)
            ## continue
            

        # vi agent
        agent = vi.valueIterAgent(N_ACTIONS, N_STATES,
                                  roadmap, skdtree, states,
                                  rewards=kwargs.get('rewards', None),
                                  gamma=gamma, T=T)
        
        # no constraints
        cstr_fns = []
        support_policy, support_values, support_validity =\
          biq.computeQ(agent, support_states,
                       support_features=(support_feature_ids,
                                         support_feature_values),
                                         support_feature_state_dict=support_feature_state_dict,
                                         cstr_fn=cstr_fns, add_no_cstr=True, max_cnt=max_cnt)

        # add normal policy
        for i in range(len(feat_range)):
            if len(support_policy_dict[i].keys())==0:
                support_policy_dict[i]   = []
                support_validity_dict[i] = []
            else:
                for j in support_policy_dict[i].keys():
                    support_policy_dict[i][j]   = np.vstack([support_policy_dict[i][j],
                                                             support_policy[j]])
                    support_validity_dict[i][j] = np.hstack([support_validity_dict[i][j],
                                                             support_validity[j]])
        support_policy_dict[-1]   = support_policy
        support_validity_dict[-1] = support_validity

        cstr_score = 0
        for idx in idx_trajs[0]:
            cstr_score += support_values[idx_trajs[0][-1]][0][idx]
        print "No cstr score = {}".format(cstr_score)
        ## from viz import viz as v
        ## v.reward_value_plot(agent.rewards, support_values[idx_trajs[0][-1]][0], states, trajs=trajs) 

        d = {}
        d['support_policy_dict']   = support_policy_dict
        d['support_validity_dict'] = support_validity_dict
        d['feat_trajs']       = feat_trajs
        d['feat_map']         = feat_map
        d['feat_range']       = feat_range
        d['cstr_feat_id']     = cstr_feat_id
        d['support_feature_ids']        = support_feature_ids
        d['support_feature_values']     = support_feature_values
        d['support_feature_state_dict'] = support_feature_state_dict
        pickle.dump( d, open( kwargs['sav_filenames']['Q'], "wb" ) )
        del support_values, support_validity_dict
        del d
    else:        
        d = pickle.load( open(kwargs['sav_filenames']['Q'], "rb"))
        support_policy_dict   = d['support_policy_dict']
        ## support_validity_dict = d['support_validity_dict']
        feat_trajs       = d['feat_trajs']
        feat_map         = d['feat_map']
        feat_range       = d['feat_range']
        support_feature_ids        = d['support_feature_ids']
        support_feature_values     = d['support_feature_values']
        support_feature_state_dict = d['support_feature_state_dict']
        cstr_feat_id     = d['cstr_feat_id']
    time1 = time.time()

        
    # initialization
    renew_log = True
    if renew_log:
        goals, z = bic.init_irl_params(n_observations, n_goals, support_policy_dict,
                                       support_states, support_feature_ids,\
                                       support_feature_state_dict,
                                       observations)        
        # log
        log = {'goals': [], 'z': [] }
        log['observations']           = observations
        log['eta']                    = eta
        log['alphas']                 = alphas
        log['support_states']         = support_states
        log['support_feature_ids']    = support_feature_ids
        log['support_feature_values'] = support_feature_values
        log['support_feature_state_dict'] = support_feature_state_dict
        ## log['support_policy_dict']    = support_policy_dict
        log['cstr_ths']               = cstr_ths
        log['cstr_feat_id']           = cstr_feat_id
    ## else:
    ##     print "Loaded saved log"
    ##     log           = pickle.load( open(kwargs['sav_filenames']['irl'], "rb"))
    ##     z             = log['z'][-1]
    ##     goals         = log['goals'][-1]
    ##     burn_in       = 0

    
    #=======================================================================================
    ## eps = np.finfo(float).eps    
    time2 = time.time()
    tqdm_e  = tqdm(range(n_iters), desc='Score', leave=True, unit=" episodes")
    for iteration in tqdm_e:
        support_validity_per_goal = []
        
        # sample subgoal & constraints 
        for j, goal in enumerate(goals):
            observation_states_part  = [s for z_i, s in zip(z, observation_states) if z_i==j]
            observation_actions_part = [a for z_i, a in zip(z, observation_actions) if z_i==j]

            # find right policy
            f              = feat_map[observation_states_part]
            f_mu           = np.mean(f, axis=0)
            f_idx          = np.argmin(np.linalg.norm(feat_range-f_mu[cstr_feat_id], axis=-1)) 
            support_policy = support_policy_dict[f_idx]
            
            goals[j] =  bic.resample_gc(observation_states_part,
                                        observation_actions_part,
                                        support_states, support_policy,
                                        alpha=alpha1,
                                        support_feature_ids=support_feature_ids,
                                        support_feature_state_dict=support_feature_state_dict,
                                        punishment=punishment,
                                        T=Ts[0],
                                        return_best=return_best,)
            goals[j][-2] = {'mu': f_mu}

        if iteration > burn_in:
            # re-ordering z and goals
            new_z, new_goals = bic.reorder(z, goals, support_states)
            # remove policies to reduce memory load
            log_goals = copy.deepcopy(new_goals)
            for i in range(len(log_goals)):
                log_goals[i][1] = None
            log['goals'].append(log_goals)
            log['z'].append(copy.deepcopy(new_z))
            
            
        #sample assignment / each observation
        tmp_use_clusters=use_clusters
        for i, obs in enumerate(observations):
            try:
                goal_state_support_ids = [support_states.index(goal[0]) for goal in goals]
            except:
                print "goal_state_support_ids is wrong"
                from IPython import embed; embed(); sys.exit()
            #reassignment
            z, goals = bic.sample_partition_assignment(obs, i, z, goals,\
                                                       support_states, support_policy_dict,
                                                       use_clusters=tmp_use_clusters, eta=eta,
                                                       alpha=alpha2,
                                                       states=states, roadmap=roadmap,
                                                       support_feature_ids=support_feature_ids,
                                                       support_feature_state_dict=support_feature_state_dict,
                                                       punishment=punishment,
                                                       T=Ts[1], enable_cstr=True,
                                                       return_best=return_best,
                                                       feat_range=feat_range)

            #post process
            if use_clusters: z, goals = bic.post_process(z, goals)
            if use_clusters is False:
                if len(goals) != n_goals: tmp_use_clusters = True
                else:                     tmp_use_clusters = False
                
        tqdm_e.set_description("t: {0:.2f}, post: {1:.2f}), goals: {2:.1f}".format(0, 0, len(goals)))
        tqdm_e.refresh()

    time3 = time.time()

    print "---------------------------------------------"
    print "Precomputation Time: {}".format(time1-time0)
    print "Gibbs Sampling Time: {}".format(time3-time2)
    print "---------------------------------------------"

    pickle.dump( log, open( kwargs['sav_filenames']['irl'], "wb" ) )    
    return log



def find_goal(mdp, env, log, states, feature_fn, roadmap,
              error=1e-10, ths=1e-3, queue_size=1000,
              enable_cstr=True, cstr_ths=2.33,
              use_discrete_state=True, use_nearest_goal=True,
              return_policy=True, **kwargs):

    irl_support_feature_ids    = log['support_feature_ids']
    irl_support_feature_values = log['support_feature_values']
    cstr_feat_id               = log['cstr_feat_id']

    # expected goals and constraints from a demonstration
    goal_features, cstr_ids, cstr_mus, _ = bic.get_expected_goal(log, states,
                                                                 enable_cstr=enable_cstr,
                                                                 queue_size=queue_size)
    
    # Now, we find goal and constraint on a new environment
    # n_cstr equals n_partitions 
    cstr_fns = make_test_cstr_fns(env, feature_fn, states, cstr_mus, cstr_feat_id, ths=cstr_ths)

    if use_discrete_state:
        T_org = copy.copy(mdp.T)

        # find feature goals
        features = fu.get_features_from_states(env, states, feature_fn)    
        distFunc = kwargs.get('distFunc', None)

        # compute q_mat per sub-goal
        new_goals = []
        for i, (f_id, c_id, c_mu) in enumerate(zip(goal_features, cstr_ids, cstr_mus)):
            print "Find {}th goal, cstr={} ".format(i, c_id)
            # feature goal
            idx  = irl_support_feature_ids.index(f_id)
            f    = irl_support_feature_values[idx]

            # get rewards
            rewards = mdp.get_rewards()
            rewards = np.array(rewards)
            rewards[np.where(rewards>0)]=0.

            # find the closest feature from goal features
            d = np.linalg.norm(features-f, ord=np.inf, axis=-1)        
            dist_ths = ths

            ## if np.amin(d) > dist_ths:
            ##     dist_ths = np.amin(d)

            #from IPython import embed; embed()#; sys.exit()            
            
            bad_goals = []
            while True:
                s_ids = [j for j in range(len(d)) if d[j] <= dist_ths]            
                if len(s_ids)>0:
                    goal_found=False
                    for idx in s_ids:
                        if idx in bad_goals: continue

                        # find a goal that violates constraints
                        if c_id==0 and cstr_fns[i](idx) is False:
                            print "Removed bad goals violating constraints"
                            print features[idx]
                            bad_goals.append(idx)
                            continue
                        
                        rx1, _ = dijkstra_planning.dijkstra_planning(env, env.start_state, states[idx],
                                                                     env.roadmap, env.states,
                                                                     distFunc=distFunc)
                        if rx1 is not None:
                            goal_found = True
                            break
                        bad_goals.append(idx)
                    print s_ids, " : Goal found? ", goal_found, " dist ths: ", dist_ths
                    if goal_found is False:
                        print "Goal feature may not match with current goal setup?"
                    if goal_found: break            
                dist_ths += ths
            print "Found goals: ", s_ids
            ## print states[s_ids]
            ## print env.get_goal_state()
            #from IPython import embed; embed()#; sys.exit()
            

            # Select the nearest state from goal and start states
            if len(s_ids)>1 and use_nearest_goal is False:
                dist = []
                for j, idx in enumerate(s_ids):
                    rx1, _ = dijkstra_planning.dijkstra_planning(env, env.start_state, states[idx],
                                                                 env.roadmap, env.states,
                                                                 distFunc=distFunc)
                    if rx1 is None:
                        dist.append(np.inf)
                        continue
                    rx2, _ = dijkstra_planning.dijkstra_planning(env, env.start_state, states[idx],
                                                                 env.roadmap, env.states,
                                                                 distFunc=distFunc)
                    if rx1 is None:
                        dist.append(np.inf)
                        continue
                    dist.append(len(rx1)+len(rx2))

                #from IPython import embed; embed(); sys.exit()
                min_j = np.argmin(dist)
                s_ids = s_ids[min_j:min_j+1]
                print "Selected a reachable state as a goal {}".format(s_ids)
            ## elif len(s_ids)>1 and use_nearest_goal:
            ##     s_

            if return_policy:

                rewards[s_ids] = 1.        
                mdp.set_rewards(rewards)

                # NOTE: we only use single constraint (0: constrained, 1: free)
                if enable_cstr is False or (cstr_fns is None or c_id>0 or c_id==-1):
                    #or (type(cstr_fns[i]) is list and c_id == len(cstr_fns[i])) \
                    #or c_id == -1:
                    # no constraint case
                    mdp.T              = copy.copy(T_org)
                else:
                    # constraint case
                    validity_map      = cstr_fns[i](range(len(states)))[roadmap]
                    validity_map[:,0] = True
                    T                 = T_org*validity_map[:,np.newaxis,:]
                    sum_T             = np.sum(T, axis=-1)
                    sum_T[np.where(sum_T==0.)] = 1.
                    T                /= sum_T[:,:,np.newaxis]            
                    mdp.T             = T

                    ## #from IPython import embed; embed()#; sys.exit()
                    ## #sys.path.insert(0,'..')
                    ## from viz import viz as v
                    ## r = cstr_fns[i](range(len(states)))
                    ## v.reward_plot(r, states)    
                    ## ## v.reward_plot_3d(r, states, env)    
                    ## #sys.exit()

                mdp.set_goal(s_ids)
                ## values, param_dict = mdp.solve_mdp(error, return_params=True)#, max_cnt=100)
                policy, values = mdp.find_policy(error)
            else:
                policy = []

            if distFunc is None:
                idx = np.argmin(np.linalg.norm(states[s_ids]-env.get_start_state(), axis=-1))
            else:
                idx = np.argmin(distFunc(states[s_ids], env.get_start_state()))
                
            if enable_cstr:
                new_goals.append([s_ids[idx], copy.copy(policy), f_id, c_mu, c_id])
            else:
                new_goals.append([s_ids[idx], copy.copy(policy), f_id])

        return new_goals
    
    else:        
        new_goals = []
        state = env.get_start_state()
        for i, (f_id, c_id, c_mu) in enumerate(zip(goal_features, cstr_ids, cstr_mus)):
            print "Find {}th goal, cstr={} ".format(i, c_id)
            # feature goal
            idx  = irl_support_feature_ids.index(f_id)
            f    = irl_support_feature_values[idx]

            if enable_cstr:
                # find the closest state from a feature f
                s = find_minimum_cost_state(state, env, f, feature_fn, cstr_feat_id,
                                            c_id, c_mu, cstr_ths)
                new_goals.append([s, None, f_id, c_mu, c_id])
            else:
                # find the closest state from a feature f
                s = find_minimum_cost_state(state, env, f, feature_fn)                
                new_goals.append([s, None, f_id])
            state =s
            
        return new_goals

    
def find_minimum_cost_state(x0, env, f, feature_fn, cstr_feat_id,
                            c_id=None, c_mu=None, c_ths=None):
    def make_objective_fn(f, feature_fn, env):
        def fn(state):
            return np.linalg.norm(feature_fn(env, state)-f) + 0.0001*np.linalg.norm(x0[:3]-state[:3])
        return fn

    fn = make_objective_fn(f, feature_fn, env)

    cons = []
    cons.append({'type': 'ineq', 'fun': lambda x: x[0]-env.observation_space.low[0] })
    cons.append({'type': 'ineq', 'fun': lambda x: x[1]-env.observation_space.low[1] })
    cons.append({'type': 'ineq', 'fun': lambda x: x[2]-env.observation_space.low[2] })

    cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[0] - x[0] })
    cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[1] - x[1] })
    cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[2] - x[2] })        
    cons.append({'type': 'eq', 'fun': lambda x: x[3:]**2 - 1. })
    
    if c_id is not None:
        for i, mu, ths in zip(cstr_feat_id, c_mu, c_ths):
            cons.append({'type': 'ineq', 'fun': lambda x: ths-abs(feature_fn(env,x)[i]-mu) })

    res = scipy.optimize.minimize(fn, x0, method='SLSQP',
                                  constraints=tuple(cons) )
    return res.x


def find_current_partition(s_idx, goals, z, alpha, eta, n_actions, states, roadmap,\
                           passed_z=[]):
    
    llh_probs = np.zeros(len(goals))
    for i, goal in enumerate(goals):
        # value
        llh_probs[i] += np.amax(bic.likelihoods([s_idx for _ in range(n_actions)],
                                                range(n_actions),
                                                goal, alpha=alpha,))
        ## llh_probs[i] += np.sum(bic.likelihoods([s_idx], [0], goal, alpha=alpha,))

    ## CRP_probs = ut.CRP(z, eta, use_clusters=False)
    prob_vector           = llh_probs #*CRP_probs
    prob_vector[passed_z] = 0.
    z_idx = np.argmax(prob_vector)

    ## # find the nearest partition given similar likelihood
    ## min_dist = np.inf
    ## min_id   = z_idx
    ## for i, p in enumerate(prob_vector):
    ##     if abs(p-prob_vector[z_idx]) < 1e-5:
    ##         if i==z_idx: continue
    ##         dist = np.linalg.norm( states[goals[i][0]]-states[s_idx] )
    ##         if min_dist >= dist:
    ##             print "found the nearest partition {}".format(i)
    ##             min_dist = dist
    ##             min_id   = i
    ## z_idx = min_id
    return z_idx

def find_action(s_idx, g_k, alpha=1.):
    """ """
    ## goal_q_mat  = g_k[1]
    ## beta  = np.exp(alpha * goal_q_mat[s_idx])
    ## beta /= np.sum(beta)
    beta = g_k[1][s_idx]
    
    max_beta = np.amax(beta)
    ids = [i for i, b in enumerate(beta) if b==max_beta]
    a_idx = random.choice(ids)    
    ## a_idx = np.argmax(beta) #/np.sum(beta))    
    return a_idx


def get_action_prior(trajs, idx_trajs, roadmap, support_states, states, support_feature_ids,
                     n_cstr=0, window_size=8):    

    actions = [[] for _ in xrange(len(support_states)) ]
    if type(trajs) is list: trajs = np.array(trajs)

    for traj, idx_traj in zip(trajs, idx_trajs):
        for i, (s, idx) in enumerate(zip(traj, idx_traj)):
            if i+1 > len(traj)-1: continue

            # get vel
            vel = []
            for j in range(-window_size/2, window_size/2):
                if i+j < 0:
                    vel.append([0,0])
                elif i+j+1>len(traj)-1:
                    vel.append([0,0])
                else:
                    vel.append( traj[i+j+1]-traj[i+j] )
                
            v1 = np.mean(vel[:len(vel)/2],axis=0) 
            v2 = np.mean(vel[len(vel)/2:],axis=0)
            a  = v2-v1

            # some actions may not be included in the observations due to the sampling in convertRawTraj
            try:
                actions[support_states.index(idx)].append( np.linalg.norm(a) )
            except:
                continue

    for i in range(len(actions)):
        if len(actions[i])==0:
            actions[i] = 0.
        else:
            actions[i] = np.mean(actions[i])
    prob_vector = actions/np.sum(actions)


    # state to feature space
    feature_prob_vector = np.zeros(max(support_feature_ids)+1)
    for p, f_id in zip(prob_vector, support_feature_ids):
        feature_prob_vector[f_id] += p

    if n_cstr>0:
        feature_prob_vector = np.repeat(feature_prob_vector, n_cstr)
    feature_prob_vector /= np.sum(feature_prob_vector)
            

    ## import matplotlib.cm as cm
    ## import matplotlib.pyplot as plt

    ## fig = plt.figure()
    ## ax = fig.add_subplot(111)
    ## poses = states[support_states]
    
    ## for i, p in enumerate(poses):
    ##     plt.plot([p[0]], [p[1]], 'ro', alpha=prob_vector[i]/np.amax(prob_vector))
    ## plt.show()
    ## sys.exit()
    
    return feature_prob_vector



## def make_cstr_fns(feature_fn, idx_trajs, feat_trajs, roadmap, states, observations,
##                   z, goals, n_actions, ths=1.):
##     """ """
##     # get partition-wise features
##     z_trajs = []
##     for idx_traj in idx_trajs:
##         z_trajs.append( ut.traj2partition(idx_traj, roadmap, states, z, observations) )
##     z_trajs    = np.array(z_trajs)
##     feat_trajs = np.array(feat_trajs)

##     n_partitions = np.amax(z_trajs)+1
##     assert len(goals)==n_partitions, "Wrong number of partitions"    
    
##     param_per_partition = [{}]*n_partitions
##     # compute mu and std
##     for i in range(n_partitions):
##         for z_traj, feat_traj in zip(z_trajs, feat_trajs):
##             ids = np.where(z_traj==i)[0]
##             mu  = np.mean(feat_traj[ids], axis=0)
##             std = np.std(feat_traj[ids], axis=0)
##         param_per_partition[i] = {'mu':mu, 'std':std }

##     eps = np.finfo(float).eps

##     def make_cstr_fn(z_idx):
##         mu  = param_per_partition[z_idx]['mu']
##         std = param_per_partition[z_idx]['std']
        
##         def cstr_fn(state_idx, f=None, state=None):
##             """cstr_fn for training"""
##             if f is None and state is None:
##                 if type(state_idx) is list or type(state_idx) is np.ndarray:
##                     f = []
##                     for s in states[state_idx]:
##                         f.append(feature_fn(s))
##                     f = np.array(f)
##                 else:
##                     f = feature_fn(states[state_idx])
##             elif state_idx is None and f is None:
##                 if type(state) is list or type(state) is np.ndarray:
##                     f = []
##                     for s in state:
##                         f.append(feature_fn(s))
##                     f = np.array(f)
##                 else:
##                     f = feature_fn(state)

##             if len(np.shape(f))==2:                
##                 ret = np.sum(np.abs(f[:,-1:]-mu[-1:]) > ths, axis=-1)
##                 ## ret = np.sum(np.abs((f[:,1:]-mu[1:])/(std[1:]+eps+1.)) > ths, axis=-1)
##             else:
##                 ret = np.sum(np.abs(f[-1:]-mu[-1:]) > ths)
##                 ## ret = np.sum(np.abs((f[1:]-mu[1:])/(std[1:]+eps+1.)) > ths)
##             return ret==0
        
##         return cstr_fn

##     return [ make_cstr_fn(i) for i in range(n_partitions) ], param_per_partition


def make_test_cstr_fns(env, feature_fn, states, cstr_mu, cstr_feat_id, ths=1.):
    """Construct a constraint function per partition
    true: valid
    """
    n_partitions = len( cstr_mu )
    features     = fu.get_features_from_states(env, states, feature_fn)

    def make_cstr_fn(z_idx):
        mu  = np.array(cstr_mu[z_idx]) # length equals number of features
        ## std = param_per_partition[z_idx]['std']
        
        def cstr_fn(state_ids):
            f = features[state_ids]
            
            if len(np.shape(f))==2:                
                ret = np.sum(np.abs(f[:,cstr_feat_id]-mu[cstr_feat_id]) > ths, axis=-1)
            else:
                ret = np.sum(np.abs(f[cstr_feat_id]-mu[cstr_feat_id]) > ths)
            return ret==0
        return cstr_fn
    
    return [ make_cstr_fn(i) for i in range(n_partitions)]


def make_train_cstr_fns(features, mu, states, cstr_feat_id, ths=1.):

    ## param_per_partition = [{}]*n_partitions
    ## # compute mu and std
    ## param_per_partition[i] = {'mu':mu} #, 'std':std }
    param = {'mu':mu}

    def cstr_fn(state_idx, f=None, state=None):
        """cstr_fn for training"""
        if f is None and state is None:
            f = features[state_idx]
        elif state_idx is None and f is None:
            ## if type(state) is list or type(state) is np.ndarray:
            ##     f = []
            ##     for s in state:
            ##         f.append(feature_fn(s))
            ##     f = np.array(f)
            ## else:
            ##     f = feature_fn(state)
            return NotImplementedError

        if len(np.shape(f))==2:                
            ret = np.sum(np.abs(f[:,cstr_feat_id]-mu[cstr_feat_id]) > ths, axis=-1)
        else:
            ret = np.sum(np.abs(f[cstr_feat_id]-mu[cstr_feat_id]) > ths)

        ## ths = 0.5
        ## if len(np.shape(f))==2:                
        ##     ret = np.sum(np.abs(f[:,-2:-1]-mu[-2:-1]) > ths, axis=-1)
        ## else:
        ##     ret = np.sum(np.abs(f[-2:-1]-mu[-2:-1]) > ths)
        return ret==0
    return [cstr_fn], param


def make_ineq_cstr_fn(feature_fn, env, mu, cstr_feat_id, ths=1., scale=1., w=1.):
    mu = np.array(mu)
        
    def cstr_fn(state):
        if len(state)==6:
            m = PyKDL.Rotation.RPY(state[3], state[4], state[5])
            quat = m.GetQuaternion()
            state = [state[0], state[1], state[2], quat[0], quat[1], quat[2], quat[3]]
            
        f   = feature_fn(env, state)
        # need to differentiable?
        return np.sum((np.abs(f[cstr_feat_id]-mu[cstr_feat_id]) - ths)*w)*scale
    return cstr_fn


def make_ineq_cstr_fns(feature_fn, env, mu, cstr_feat_id, ths=1., w=1., scale=1.):
    mu = np.array(mu)
        
    def cstr_fns(state):
        if len(state)==6:
            m = PyKDL.Rotation.RPY(state[3], state[4], state[5])
            quat = m.GetQuaternion()
            state = [state[0], state[1], state[2], quat[0], quat[1], quat[2], quat[3]]
            
        f   = feature_fn(env, state)
        # need to differentiable?
        return (np.abs(f[cstr_feat_id]-mu[cstr_feat_id]) - ths)*w*scale
    return cstr_fns


def make_cost_fn(feature_fn, env, mu, cstr_feat_id, scale=1., w=1.):
    def cost_fn(state):
        if len(state)==6:
            m = PyKDL.Rotation.RPY(state[3], state[4], state[5])
            quat = m.GetQuaternion()
            state = [state[0], state[1], state[2], quat[0], quat[1], quat[2], quat[3]]
            
        f   = feature_fn(env, state)
        # need to differentiable?
        return np.sum(np.abs(f[cstr_feat_id]-mu[cstr_feat_id])*w)*scale
    return cost_fn
    


def viz_feat_traj(feat_traj, save_plot=False):

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    from matplotlib import rc
    #rc('font',**{'family':'serif','sans-serif':['Times New Roman']})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 16})
    rc('text', usetex=True)


    fig = plt.figure()
    for i in range(len(feat_traj[0])):
        ax = fig.add_subplot(len(feat_traj[0]),1,i+1)
        plt.plot(feat_traj[:,i], linewidth=2)
        ax.set_ylabel(r'$f_{}$'.format(i+1))
        ax.set_ylim( [-0.1, 1.1] )
        plt.xticks([], [])

    ax.set_xlabel('Time steps')
    #plt.axis('off')
    plt.xticks([], [])


    if save_plot:
        fig.savefig("test.png", format='png')
        fig.savefig("test.pdf", format='pdf')
    else:
        plt.show()
    
    sys.exit()
    return
