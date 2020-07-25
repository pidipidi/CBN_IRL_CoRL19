import sys, os, copy, time
import random
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import dirichlet, trim_mean

## sys.path.insert(0,'..')
#from mdp import parallel_value_iteration as vi
from mdp import value_iteration as vi
#from mdp import constrained_value_iteration as vi
from mdp import feature_utils as fu
import bn_irl_utils as ut
import bn_irl_common as bic
import bn_irl_q_fn as biq

from cbn_irl.path_planning import dijkstra_planning 


def bn_irl(env, roadmap, skdtree, states, T, gamma, trajs, idx_trajs,
           feature_fn, alphas=(0.1,1.0), eta=0.5, punishment=0., Ts=[0.1, 0.7], **kwargs):
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
    n_goals = kwargs.get('n_goals', 2)
    burn_in = kwargs.get('burn_in', 1000)
    n_iters = kwargs.get('n_iter', 2000)
    n_iters = n_iters if n_iters > burn_in else int(n_iters*1.5)
    use_clusters     = True
    use_action_prior = kwargs.get('use_action_prior', False)
    alpha1, alpha2   = alphas
    max_cnt = kwargs.get('max_cnt', 100)
    return_best = False
    
    N_STATES  = len(states)
    N_ACTIONS = len(roadmap[0])

    # support observations (i.e., demonstrations)
    observations     = ut.trajs2observations(trajs, idx_trajs, roadmap, states)
    support_states   = bic.get_support_space(observations)
    n_observations   = len(observations)
    n_support_states = len(support_states)

    observation_states  = [obs.state for obs in observations]
    observation_actions = [obs.action for obs in observations]
    
    # vi agent
    agent = vi.valueIterAgent(N_ACTIONS, N_STATES,
                              roadmap, skdtree, states,
                              rewards=kwargs.get('rewards', None),
                              gamma=gamma, T=T)

    # precomputation Q and pi per support features that is a list of goals [feature, policy]
    time0 = time.time()
    support_values = None
    support_validity = None
    if os.path.isfile(kwargs['sav_filenames']['Q']) is False or kwargs.get('vi_renew', False):
        assert len(np.shape(trajs))==3, "wrong shape or number of trajectories"
        print ("Renewing q functions")

        feat_map = fu.get_features_from_states(env, states, feature_fn)
        # get features given each feature sub goal
        support_feature_ids, support_feature_values = bic.get_support_feature_space(support_states,
                                                                                    states, feat_map,
                                                                                    env)
        support_feature_state_dict = bic.get_feature_to_state_dict(support_states, support_feature_ids)
        
        support_policy, support_values =\
        biq.computeQ(agent, support_states,
                     support_features=(support_feature_ids,
                                       support_feature_values),
                                       support_feature_state_dict=support_feature_state_dict,
                                       max_cnt=max_cnt)
        
        d = {}
        d['support_policy'] = support_policy
        d['support_feature_ids']        = support_feature_ids
        d['support_feature_values']     = support_feature_values
        d['support_feature_state_dict'] = support_feature_state_dict
        pickle.dump( d, open( kwargs['sav_filenames']['Q'], "wb" ) )
    else:        
        d = pickle.load( open(kwargs['sav_filenames']['Q'], "rb"))
        support_policy = d['support_policy']
        support_feature_ids        = d['support_feature_ids']
        support_feature_values     = d['support_feature_values']
        support_feature_state_dict = d['support_feature_state_dict']
    time1 = time.time()

 
    # initialization
    renew_log = True
    if renew_log:
        goals, z = bic.init_irl_params(n_observations, n_goals, support_policy,
                                       support_states, support_feature_ids,\
                                       support_feature_state_dict,
                                       observations)        
        # log
        log = {'goals': [], 'z': []}
        log['observations']           = observations
        log['eta']                    = eta
        log['alphas']                 = alphas
        log['support_states']         = support_states
        log['support_feature_ids']    = support_feature_ids
        log['support_feature_values'] = support_feature_values
        log['support_feature_state_dict'] = support_feature_state_dict
    else:
        print "Loaded saved log"
        log           = pickle.load( open(kwargs['sav_filenames']['irl'], "rb"))
        z             = log['z'][-1]
        goals         = log['goals'][-1]
        burn_in       = 0

    #=======================================================================================
    ## eps = np.finfo(float).eps

    time2 = time.time()
    tqdm_e  = tqdm(range(n_iters), desc='Score', leave=True, unit=" episodes")
    for iteration in tqdm_e: 
        # sample subgoal & constraints 
        for j, goal in enumerate(goals):
            observation_states_part  = [s for z_i, s in zip(z, observation_states) if z_i==j]
            observation_actions_part = [a for z_i, a in zip(z, observation_actions) if z_i==j]
            
            goals[j] =  bic.resample(observation_states_part, observation_actions_part,
                                     support_states, support_policy, #prior=prior,
                                     alpha=alpha1,
                                     support_feature_ids=support_feature_ids,
                                     support_feature_state_dict=support_feature_state_dict,
                                     T=Ts[0],
                                     punishment=punishment,
                                     return_best=return_best,)


        if iteration > burn_in:            
            # re-ordering z and goals
            new_z, new_goals = bic.reorder(z, goals, support_states)
            # remove policies to reduce memory load
            log_goals = copy.deepcopy(new_goals)
            for i in range(len(log_goals)):
                log_goals[i][1] = None
            log['goals'].append(log_goals)
            log['z'].append(copy.deepcopy(new_z))
            ## if iteration%500==0:
            ##     pickle.dump( log, open( kwargs['sav_filenames']['irl'], "wb" ) )
                                     
           
        # sample assignment / each observation
        tmp_use_clusters=use_clusters
        for i, obs in enumerate(observations):
            goal_state_support_ids = [support_states.index(goal[0]) for goal in goals]
            #reassignment
            z, goals = bic.sample_partition_assignment(obs, i, z, goals,\
                                                       support_states, support_policy,
                                                       use_clusters=tmp_use_clusters, eta=eta,
                                                       alpha=alpha2,
                                                       states=states, roadmap=roadmap,
                                                       support_feature_ids=support_feature_ids,
                                                       support_feature_state_dict=support_feature_state_dict,
                                                       punishment=punishment,
                                                       T=Ts[1],
                                                       return_best=return_best,)

            #post process
            if use_clusters: z, goals = bic.post_process(z, goals)
            if use_clusters is False:
                if len(goals) != n_goals: tmp_use_clusters = True
                else:                     tmp_use_clusters = False

        tqdm_e.set_description("goals: {0:.1f}".format(len(goals)))
        tqdm_e.refresh()

    time3 = time.time()

    print "---------------------------------------------"
    print "Precomputation Time: {}".format(time1-time0)
    print "Gibbs Sampling Time: {}".format(time3-time2)
    print "---------------------------------------------"
    
    pickle.dump( log, open( kwargs['sav_filenames']['irl'], "wb" ) )    
    return log
    

def find_goal(mdp, env, log, states, feature_fn, cstr_fn=None, error=1e-10, ths=1e-3,\
              queue_size=1000, use_nearest_goal=True, **kwargs):

    irl_support_feature_ids    = log['support_feature_ids']
    irl_support_feature_values = log['support_feature_values']

    goal_features, _, _, _ = bic.get_expected_goal(log, states, queue_size=queue_size)
    T_org = copy.copy(mdp.T)

    # find feature goals
    features = fu.get_features_from_states(env, states, feature_fn)    

    distFunc = kwargs.get('distFunc', None)

    # compute q_mat for a sub-goal
    new_goals = []
    for i, f_id in enumerate(goal_features):
        print "-------------------------------------------------------------------"
        print "Find {}th goal".format(i)
        # feature goal
        idx  = irl_support_feature_ids.index(f_id)
        f    = irl_support_feature_values[idx]
       
        # get rewards
        rewards = mdp.get_rewards()
        rewards = np.array(rewards)
        rewards[np.where(rewards>0)]=0.

        ## from IPython import embed; embed(); sys.exit()
        # find the closest state from a goal
        d = np.linalg.norm(features-f, ord=np.inf, axis=-1)        
        dist_ths = ths

        if np.amin(d) > dist_ths:
            dist_ths = np.amin(d)
        
        bad_goals = []
        while True:
            s_ids = [j for j in range(len(d)) if d[j] <= dist_ths and (features[j]>0).all()]            
            if len(s_ids)>0:
                goal_found=False
                for idx in s_ids:
                    if idx in bad_goals: continue
                    rx1, _ = dijkstra_planning.dijkstra_planning(env, env.start_state, states[idx],
                                                                 env.roadmap, env.states,
                                                                 distFunc=distFunc)
                    if rx1 is not None:
                        goal_found = True
                        break
                    bad_goals.append(idx)                
                if goal_found:
                    print s_ids, goal_found, dist_ths                    
                    break
            print "Increase the goal range"
            dist_ths += ths
        print "----------------------------"
        print "Found sub-goals: ", s_ids
        print "----------------------------", len(s_ids)

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
                rx2, _ = dijkstra_planning.dijkstra_planning(env, states[idx], env.goal_state,
                                                             env.roadmap, env.states,
                                                             distFunc=distFunc)
                if rx2 is None:
                    dist.append(np.inf)
                    continue
                dist.append(len(rx1)+len(rx2))

            #from IPython import embed; embed(); sys.exit()
            min_j = np.argmin(dist)
            s_ids = s_ids[min_j:min_j+1]
            print "Selected a reachable state as a goal {}".format(s_ids)

        # set new rewards
        rewards[s_ids] = 1.
        mdp.set_rewards(rewards)

        #
        print "Start solve policy with new reward and T"
        mdp.T          = copy.copy(T_org)
        ## policy, values = mdp.find_policy(error)
        ## new_goals.append([s_ids[0], copy.copy(policy), f_id])
        new_goals.append([s_ids[0], None, f_id])


    goal_state_ids = []
    for i in range(len(new_goals)):
        goal_state_ids.append( new_goals[i][0] )
    policies = biq.computePolicies(mdp, goal_state_ids, error)
    for i in range(len(new_goals)):
        new_goals[i][1] = policies[i]
    
        

    return new_goals


def find_current_partition(s_idx, goals, z, alpha, eta, n_actions, states, roadmap, passed_z=[],
                           cstr_fn=None):

    ## CRP_probs = ut.CRP(z, eta, use_clusters=False)
    
    #from IPython import embed; embed(); sys.exit()
    llh_probs = np.zeros(len(goals))
    for i, goal in enumerate(goals):
        # value
        llh_probs[i] += np.amax(bic.likelihoods([s_idx for _ in range(n_actions)],
                                                range(n_actions),
                                                goal, alpha=alpha,))

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
    beta = g_k[1][s_idx]
    
    max_beta = np.amax(beta)
    ids = [i for i, b in enumerate(beta) if b==max_beta]
    a_idx = random.choice(ids)    
    ## a_idx = np.argmax(beta) #/np.sum(beta))
    return a_idx


def get_action_prior(trajs, idx_trajs, roadmap, support_states, states, support_feature_ids,
                     n_cstr=0):

    window_size = 8

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
            a  = v2-v1 # need metric?

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


def visualization(env, z, goals, observations, states, support_states, support_feature_ids, trajs=None,
                  alpha=1., punishment=5., q_mat=None):

    from mdp.bn_irl import bn_irl_common as bic
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    ## fig = plt.figure(figsize=(10,10))
    ## fig = plt.figure()
    fig, ax = plt.subplots()


    ## if q_mat is not None:
    ##     img = np.zeros((60,60))
    ##     for q, s in zip(q_mat, states):
    ##         print q
    ##         img[int(s[0]), int(s[1])] += np.sum(q)
    ##     ## img -= np.amin(img)
    ##     ## img /= np.amax(img)
    ##     ## mag = 0.1
    ##     ## img = np.exp(mag*img)/np.sum(np.exp(mag*img))
    ##     plt.imshow(img.T, origin='lower', interpolation='nearest')

        
    if trajs is not None:
        for traj in trajs:
            plt.plot(np.array(traj)[:,0], np.array(traj)[:,1], 'r-', alpha=0.2)
    ## plt.show()
    ## sys.exit()
    
    plt.plot(env.objects[:,0], env.objects[:,1], 'ko')

    occurrence = ut.tally(z)
    colors = cm.rainbow(np.linspace(0, 1, len(occurrence)))
    ## print len(occurrence), len(goals)
    for i in range(len(occurrence)):
        ids = [j for j, v in enumerate(z) if v==i]
        obs_zi = np.array(observations)[ids]
        if len(obs_zi)<1: continue
        obs  = np.array([states[obs_zi[j].state] for j in range(len(obs_zi))])

        a=[]
        for o in obs_zi:
            a.append( bic.likelihood(o, goals[i], alpha=alpha, normalization=False, punishment=punishment) )
        a = np.array(a)-min(a)
        a = np.array(a)/max(a)
        
        for j, o in enumerate(obs):
            plt.plot([o[0]], [o[1]], 'o', c=colors[i], markersize=5, alpha=0.2)#, alpha=a[j])
        ## plt.scatter(obs[:,0], obs[:,1], c=colors[i])

        goal_states = bic.get_state_goal_from_feature(goals[i][2], support_states, support_feature_ids)
        goal_states = states[goal_states]
        plt.scatter(goal_states[:,0], goal_states[:,1], marker='P', c=colors[i], s=400)
        #plt.text(goal_states[:,0], goal_states[:,1]-5, str(i) )

    plt.text(env.start_state[0]-3, env.start_state[1]-5, "Start", fontsize=16 )
    plt.text(env.goal_state[0]-3, env.goal_state[1]-5, "Goal", fontsize=16 )

    lines  = [plt.scatter(np.linspace(-10,-5,len(colors)), np.linspace(-10,-5,len(colors)),\
                          marker='P', c=colors, s=50),
              plt.scatter(np.linspace(-10,-5,len(colors)), np.linspace(-10,-5,len(colors)),\
                       marker='o', c=colors[i], s=50, alpha=0.2)]
    labels = ['Sub-goal', 'Partition'] 
    plt.legend(lines, labels, fontsize='large', loc=1)

    plt.xlim(env.observation_space.low[0], env.observation_space.high[0] )
    plt.ylim(env.observation_space.low[1], env.observation_space.high[1] )
    plt.xticks([]),plt.yticks([])
    
    plt.show()


