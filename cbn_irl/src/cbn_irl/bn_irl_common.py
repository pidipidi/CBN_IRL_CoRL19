import sys, os, copy
import random
import pickle
import numpy as np
from tqdm import tqdm

from scipy.stats import multinomial
import collections   
import rospkg

## sys.path.insert(0,'..')
from mdp import parallel_value_iteration as vi
import bn_irl_utils as ut

eps = np.finfo(np.float).eps

class O:
    def __init__(self, state, action):
        self.state  = state
        self.action = action


def get_support_space(observations):
    """Get unique observations"""
    states = []
    for i, obs in enumerate(observations):
        states.append(obs.state)

    # NOTE: there can be same state with different actions.........
    #_, indices = np.unique(states, return_index=True)

    unique_states = []
    for i, obs in enumerate(observations):        
        if not(obs.state in unique_states):
            unique_states.append(obs.state)

    return unique_states


def get_support_feature_space(support_states, states, features, env, ths=0):
    """Return each support state's feature index in a unique feature vector"""
    indices = []
    unique_features = []
    
    for s in support_states:
        ## f = feature_fn(env, states[s]) # need to parallelize!
        f = features[s]
        if len(unique_features)==0:
            unique_features.append(f)
            indices.append(0)
        else:
            d = np.linalg.norm(unique_features-f, ord=np.inf, axis=-1)
                
            if min(d) > ths:
                unique_features.append(f)
                indices.append(len(unique_features)-1)                
            else:
                indices.append(np.argmin(d))


    return indices, unique_features



def get_feature_to_state_dict(support_states, support_feature_ids):

    d = {}
    max_f_idx = max(support_feature_ids)
    for feature_id in range(max_f_idx+1):
        ids = [idx for idx, f_idx in enumerate(support_feature_ids) if f_idx == feature_id]
        d[feature_id] = np.array(support_states)[ids]        
    return d

    

def likelihood(observation, goal, alpha=1., punishment=0., min_val=1e-10,
               observation_goal_action=None, **kwargs):
    """
    return a list of P(a_i |  s_i, z_j)
    """
    ## goal_state = goal[0]
    ## if observation.state == goal_state:
    ##     return min_val
    
    ## beta = np.exp(1./alpha * goal_policy[observation.state])
    ## tmp  =  1./alpha*goal_policy[observation.state, observation.action] - np.log(np.sum(beta)) 
    ## tmp  = np.exp(tmp)
    ## ## if tmp > min_val: return tmp
    ## return tmp

    # control based
    ## states  = kwargs['states']
    ## roadmap = kwargs['roadmap']
    ## tgt = roadmap[observation.state][observation.action]
    ## ## a_des = states[goal_state]-states[observation.state]
    ## ## a     = states[tgt]-states[observation.state]
    ## a_des = states[goal_state]
    ## a     = states[tgt]
    ## return np.exp(-alpha*np.linalg.norm(a_des-a))

    # paper
    ## tmp = np.exp(alpha * goal[1][observation.state, observation.action])
    ## if kwargs.get('normalization', True) or True:
    ##     tmp /= np.sum(np.exp(alpha * goal[1][observation.state]))

    ## beta = np.exp(alpha * goal[1][observation.state, observation.action])
    beta = np.exp(alpha * goal[1][observation.state])
    #beta /= np.sum(beta+eps)
    #beta *= (beta-np.amin(beta))/(np.amax(beta)-np.amin(beta) + eps)
    #beta /= np.sum(beta+eps)
    beta = beta[observation.action]+eps

    
    ## if observation_goal_action is not None:
    ##     return beta * observation_goal_action
    return beta
    
    ## ## tmp  = beta[observation.action] - punishment * (np.amax(beta)-beta[observation.action])
    ## r    = (beta[observation.action]-np.amin(beta))/(np.amax(beta)-np.amin(beta) + 10*eps)
    ## tmp  = beta[observation.action] * (punishment+r)/(punishment+1.)

    ## ## tmp  = beta[observation.action] * (1. - punishment * (np.amax(beta)-beta[observation.action]))
    ## ## if tmp<0.: tmp=0.
    ## ## return tmp
    ## return tmp
    ## ## if tmp > min_val: return tmp
    ## ## return min_val


def likelihoods(observation_states, observation_actions, goal, alpha=1.,
                punishment=0., min_val=1e-10, observation_goal_actions=None, **kwargs):

    beta  = np.exp(alpha * goal[1][observation_states])
    beta /= (np.sum(beta, axis=-1)[:, np.newaxis]+eps)

    ## beta_max = np.amax(beta, axis=-1)[:, np.newaxis]
    ## beta_min = np.amin(beta, axis=-1)[:, np.newaxis]
    ## r        = (beta-beta_min)/(beta_max-beta_min+eps)
    
    #punishment = 0.
    #beta *= (punishment+r)/(punishment+1.)
    ## beta  = beta**1
    #beta /= np.sum(beta, axis=-1)[:, np.newaxis]
    ## beta_sum = np.sum(beta, axis=-1)
    ## beta_sum[np.where(beta_sum==0)] = 1.
    ## beta /= beta_sum[:, np.newaxis]
    ## from IPython import embed; embed(); sys.exit()
    l = beta[range(len(observation_actions)),observation_actions]
    return l

    

    #l *= 1.-punishment*(np.amax(beta, axis=-1)-l)
    ## #l /= np.sum(beta, axis=-1)    
    #l[l<0] = 0.
    ## return l #* observation_goal_actions

    #from IPython import embed; embed(); sys.exit()

    ## if observation_goal_actions is not None:
    ##     return l * observation_goal_actions
    ## else:
    ## return l
    
    ## elif True:
    ##     ## l = beta[:,observation_actions]
    ##     beta_max = np.amax(beta, axis=-1)
    ##     beta_min = np.amin(beta, axis=-1)
    ##     r   = (l-beta_min) / (beta_max-beta_min + 10*eps)    # 0~1 scale
    ##     return l * (punishment+r)/(punishment+1.)
        
    ## return tmp
        

def likelihood_vector(observation_states, observation_actions, support_policy,
                      support_states, alpha=1.,
                      punishment=0., support_feature_ids=None,
                      support_feature_state_dict=None,
                      observation_goal_actions=None, **kwargs):
    """
    support states : goals in a partition 
    """
    if support_feature_ids is None:
        llh_vector = np.zeros(len(support_states))
        
        # likelihood per= each potential goal (in a partition)
        for i, goal_state in enumerate(support_states):
            llh_vector[i] = np.sum(likelihoods(observation_states, observation_actions, \
                                               [goal_state, support_policy[goal_state]],
                                               alpha=alpha,
                                               punishment=punishment))
    else:
        n_unique_feature = max(support_feature_ids)+1    
        llh_vector = np.zeros(n_unique_feature)
        
        # likelihood per each potential goal (in a partition)
        tmp = []
        for idx, i in enumerate(support_feature_ids):
            if i in tmp: continue
            tmp.append(i)

            if observation_goal_actions is not None:
                goal_actions = observation_goal_actions[:,idx]
            else:
                goal_actions = None
            
            goal_states   = support_feature_state_dict[i]
            llh_vector[i] = np.sum(likelihoods(observation_states, observation_actions, \
                                               [goal_states[0], support_policy[goal_states[0]]],
                                               alpha=alpha,
                                               punishment=punishment,
                                               observation_goal_actions=goal_actions))
            
    return llh_vector

def likelihood_vector_gc(observation_states, observation_actions, support_policy,
                         support_states, support_feature_state_dict, alpha=1.,
                         punishment=0., support_feature_ids=None,
                         support_validity=None, observation_goal_actions=None):
    """
    support states : goals in a partition 
    """
    if support_feature_ids is None:
        return NotImplementedError
    else:
        n_unique_feature = max(support_feature_ids)+1
        n_cstr           = len(support_policy[ support_policy.keys()[0] ])
        llh_vector       = np.zeros(n_unique_feature*n_cstr)

        # likelihood per each potential goal (in a partition)
        tmp = []
        for idx, i in enumerate(support_feature_ids):
            if i in tmp: continue
            tmp.append(i)

            if observation_goal_actions is not None:
                goal_actions = observation_goal_actions[:,idx]
            else:
                goal_actions = None

            goal_states = support_feature_state_dict[i]
            for j in range(n_cstr):

                llh_vector[i*n_cstr+j] = np.sum(likelihoods(observation_states, observation_actions, \
                                                            [goal_states[-1],\
                                                             support_policy[goal_states[-1]][j]],\
                                                             ## support_policy[goal_states[-1]].astype(np.float)[j]],\
                                                             alpha=alpha,\
                                                             punishment=punishment,\
                                                             observation_goal_actions=goal_actions))
    return llh_vector


def resample(observation_states, observation_actions, support_states, support_policy,
             prior=None, alpha=1., punishment=0., support_feature_ids=None,
             support_feature_state_dict=None, observation_goal_actions=None,
             T=1., return_best=False, **kwargs):
    """
    \sum p(O_i | g_z_i) x p(g_j)
    """
    # prob_vector over support_states or feature ids
    prob_vector  = likelihood_vector(observation_states, observation_actions,
                                     support_policy, \
                                     support_states, alpha=alpha, punishment=punishment,
                                     support_feature_ids=support_feature_ids,
                                     support_feature_state_dict=support_feature_state_dict,
                                     observation_goal_actions=observation_goal_actions,)
    
    if prior is not None: prob_vector *= prior
       
    if return_best:
        chosen        = np.argmax(prob_vector)
    else:
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_vector  = prob_vector**T 
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_sum    = np.sum(prob_vector)

        while prob_sum < 0.99:
            prob_vector /= (np.sum(prob_vector)+eps)
            prob_sum     = np.sum(prob_vector)                      
        try:
            assert prob_sum>=0.95, "prob sum {} is lower than 1.".format(prob_sum)
        except:
            from IPython import embed; embed(); sys.exit()
        rv            = multinomial(n=1,p=prob_vector)
        chosen        = np.argmax(rv.rvs(1))

    if support_feature_ids is None:
        goal_chosen = support_states[chosen]
        policy_chosen = support_policy[goal_chosen]
        return [goal_chosen, policy_chosen]
    else:
        #chosen is feature id
        goal_chosen = support_feature_state_dict[chosen]            
        policy_chosen = support_policy[goal_chosen[0]]
        return [goal_chosen[0], policy_chosen, chosen]


def resample_gc(observation_states, observation_actions, support_states, support_policy,
                prior=None, alpha=1., punishment=0., support_feature_ids=None,
                support_feature_state_dict=None, support_validity=None,
                observation_goal_actions=None,
                T=1., partition_id=None, return_best=False):
    """
    \sum p(O_i | g_z_i, c_z_i) x p(g_j, c_j)
    """
    # prob_vector over (support_states or feature ids) x constraint index
    prob_vector  = likelihood_vector_gc(observation_states, observation_actions,
                                        support_policy, \
                                        support_states,\
                                        support_feature_state_dict,\
                                        alpha=alpha, punishment=punishment,
                                        support_feature_ids=support_feature_ids,
                                        support_validity=support_validity,
                                        observation_goal_actions=observation_goal_actions)
    if prior is not None: prob_vector *= prior
    
    # sampling
    if return_best:
        chosen = np.argmax(prob_vector)
    else:
        ## ids = prob_vector.argsort()[:int(len(prob_vector)*0.2)] #hack?
        ## prob_vector[ids] = 0.
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_vector  = prob_vector**T #(1./T)
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_sum     = np.sum(prob_vector)

        while prob_sum < 0.99:
            prob_vector /= (np.sum(prob_vector)+eps)
            prob_sum     = np.sum(prob_vector)                      
        try:
            assert prob_sum>=0.95, "prob sum {} is lower than 1.".format(prob_sum)
        except:
            from IPython import embed; embed(); sys.exit()
            
        rv            = multinomial(n=1,p=prob_vector)
        chosen        = np.argmax(rv.rvs(1))

    ## # Debug Print for cstr
    ## goal_array = np.reshape(prob_vector, (-1,2))
    ## c_idx = np.argmax(goal_array[:,0])
    ## g_idx = np.argmax(goal_array[:,1])
    ## # Constraint-based goal's probability
    ## print goal_array[c_idx][0]/sum(goal_array[c_idx]), goal_array[c_idx], "  00000000000  ", 
    ## # Normal goal's probability
    ## print goal_array[g_idx][1]/sum(goal_array[g_idx]), "  00000000000  ", g_idx, np.argmax(goal_array[g_idx])
    ## ## from IPython import embed; embed(); sys.exit()
        
    if support_feature_ids is None:
        return NotImplementedError
    else:
        #chosen is feature id
        feat_chosen = chosen / len(support_policy[ support_policy.keys()[0] ])
        cstr_chosen = chosen % len(support_policy[ support_policy.keys()[0] ])

        goal_chosen = support_feature_state_dict[feat_chosen]            
        policy_chosen = support_policy[goal_chosen[0]][cstr_chosen]
        
        if not (goal_chosen in support_states):
            print "chosen goal is not in support_states"
            from IPython import embed; embed(); sys.exit()
        ## best_chosen = np.argmax(prob_vector)
        ## best_feat_chosen = best_chosen / len(support_policy[ support_policy.keys()[0] ])
        ## best_goal_chosen = support_feature_state_dict[best_feat_chosen]            
        ## print support_states.index(goal_chosen[0]), support_states.index(best_goal_chosen[0]) 
        return [goal_chosen[0], policy_chosen, feat_chosen, None, cstr_chosen]


def sample(support_states, support_policy, support_feature_ids=None, roadmap=None, z=None,
           goal_state=None, passed_z=[], support_feature_state_dict=None):
    if support_feature_ids is None:
        if roadmap is not None and z is not None and goal_state is not None:
            return NotImplementedError
        else:
            state = random.choice(support_states)
            
        return [state, support_policy[state].astype(np.float)]
    else:
        f_id = random.choice(np.unique(support_feature_ids))
        states = support_feature_state_dict[f_id]            
        return [states[0], support_policy[states[0]].astype(np.float), f_id]


def sample_gc(support_states, support_policy, support_feature_ids=None, roadmap=None, z=None,
              support_feature_state_dict=None, feat_range=None):

    if type(support_policy[support_policy.keys()[0]]) is dict:
        if feat_range is None: f_idx = -1
        else:                  f_idx = random.choice(range(len(feat_range)))
        support_policy_dict = support_policy
        support_policy      = support_policy_dict[f_idx] 
    else:
        support_policy_dict = None
        
    #from IPython import embed; embed(); sys.exit()
    
    ## cstr_idx = -1 #np.random.randint(len(support_policy[ support_policy.keys()[0] ]))
    cstr_idx = random.choice(range(len(support_policy[ support_policy.keys()[0] ])))
    goal     = sample(support_states, support_policy, support_feature_ids=support_feature_ids,
                      support_feature_state_dict=support_feature_state_dict)
    goal[1]  = goal[1][cstr_idx]

    #temp
    ## return goal + [{'mu': feat_range[f_idx]}, cstr_idx ]
    return goal + [None, cstr_idx ]


def is_boundary_state(state, support_states, z, roadmap, passed_z=[], return_count=False):
    """Is the state is on the boundary of the partitions?"""
    s_idx = support_states.index(state)
    cur_z = z[s_idx]
                
    next_states = roadmap[state]
    count = 0
    
    for s in next_states:
        if s in support_states:
            idx = support_states.index(s)
        else:
            continue
        nxt_z = z[idx]

        if return_count:
            if cur_z != nxt_z:
                count += 1
        else:
            if cur_z != nxt_z:
                return True    

    if return_count:
        if count > 0: return True, count
        else: return False, count
        
    return False
    
    

def sample_partition_assignment(obs, obs_i, z, goals,
                                support_states, support_policy,
                                use_clusters=True, eta=0.5,
                                alpha=1., punishment=0., states=None, roadmap=None,
                                support_feature_ids=None, goal_state=None,
                                support_feature_state_dict=None,
                                return_best=False, support_validity=None, T=1.,
                                enable_cstr=False, observation_goal_action=None,
                                feat_range=None, **kwargs):

    # Get the CRP probabilities
    CRP_probs = ut.CRP(z, eta, use_clusters=use_clusters) 
    prob_size = len(goals)+1 if use_clusters else len(goals)
    llh_probs = np.zeros(prob_size)

    # Calculate likelihood of observation per goal
    for j,g in enumerate(goals):
        ## if support_validity is not None and j < len(support_validity):
        ##     s       = g[0]
        ##     cstr_id = g[-1]
        ##     validity_mask = support_validity[j][s][cstr_id]*1.
        ## else:
        ##     validity_mask = 1.
        llh_probs[j] = likelihood(obs, g, alpha=alpha,)
                                  ## observation_goal_action=observation_goal_action[j])
                                  ## * validity_mask

    if use_clusters:
        # Sample a potential new goal
        if enable_cstr is False:
            potential_g = sample(support_states, support_policy, \
                                 support_feature_ids=support_feature_ids,
                                 support_feature_state_dict=support_feature_state_dict)
        else:
            potential_g = sample_gc(support_states, \
                                    support_policy, support_feature_ids=support_feature_ids,
                                    support_feature_state_dict=support_feature_state_dict,
                                    feat_range=feat_range)
            
        llh_probs[-1] = likelihood(obs, potential_g, alpha=alpha)
        
    assert len(CRP_probs) == len(llh_probs), \
      "{} CRP_probs and {} llh_probs are different.".format(len(CRP_probs), len(llh_probs))
    
    # Put probabilities together and normalise
    prob_vector  = llh_probs * CRP_probs
       
    # Sample new assignement
    if return_best:
        chosen   = np.argmax(prob_vector)
    else:
        ## ids = prob_vector.argsort()[:int(len(prob_vector)*0.2)]
        ## prob_vector[ids] = 0.
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_vector = prob_vector**T
        prob_vector /= (np.sum(prob_vector)+eps)
        prob_sum    = np.sum(prob_vector)
        assert prob_sum>=0.98, "assignment prob sum {} is lower than 1.".format(prob_sum)
        rv            = multinomial(n=1,p=prob_vector)
        chosen        = np.argmax(rv.rvs(1))

    z[obs_i] = chosen
    if chosen == len(goals) and use_clusters:
        goals.append(potential_g)
    return z, goals


def post_process(z, goals):
    tally_z = ut.tally(z)

    if len(goals)!=len(tally_z):
        goals = goals[:-1] #[g for i, g in enumerate(goals) if i<len(goals)-1]
    
    for i in reversed(range(len(tally_z))):
        if tally_z[i] == 0:
            for j, p in enumerate(z):
                if p>i: z[j]-=1
            goals = [g for j, g in enumerate(goals) if j!=i]

    ## if len(goals) != len(ut.tally(z)):
    ##     goals = goals[:-1] #[g for j, g in enumerate(goals) if j!=len(goals)-1]
            
    assert len(goals)==len(ut.tally(z)), "number of partitions {} and goals {} are different".format(len(ut.tally(z)),len(goals))
    
    return z, goals


def reorder(z, goals, support_states):
    """Change the order of goals and partitions"""

    # indices
    ids = []
    for goal in goals:
        ids.append( support_states.index(goal[0]) )

    ids = np.argsort(ids)

    new_z = copy.copy(z)
    for i in range(len(ids)):
        new_z[z==ids[i]] = i
    new_goals = [goals[i] for i in ids]
    return new_z, new_goals


def partitioning_loss(goals, observations, z):
    loss = 0
    for i, obs in enumerate(observations):
        goal = goals[z[i]]
        policy_action = np.argmax(goal[1][obs.state])
        if policy_action != obs.action:
            loss += 1.
    return loss


def get_posterior_prob(z, goal, observations, alpha=1., punishment=0., states=None):
    posterior = 0.
    for j, obs in enumerate(observations):

        g = goal[z[j]]
        posterior += likelihood(obs, g, alpha=alpha, normalization=False, punishment=punishment)

    return posterior

## def get_empirical_mode(goals, partitions, observations, alpha=1., punishment=5.):
##     """
##     Find a max as a mode in the multimodal distribution of posterior    
##     """
##     probs = []
##     for i, (z, goal) in enumerate(zip(partitions, goals)):
##         posterior = get_posterior_prob(z, goal, observations, alpha, punishment=punishment)
##         ## loss = bic.partitioning_loss(g, observations, z)
##         probs.append(posterior)

##     i = np.argmax(probs)
##     print "Max posterior: {} and Min posterior {}".format(max(probs), min(probs))
        
##     return partitions[i], goals[i]


def get_expected_goal(log, states, queue_size=1000, enable_cstr=False, idx_traj=None,
                      return_params=False):
    """Return a set of expected goal features and constraints
    """
    idx_traj = log['support_states']
    
    # 1. expected number of goals and expected feature ids
    n_g_queue = collections.deque(maxlen=queue_size)    
    for i, goals in enumerate(log['goals'][-queue_size:]):
        n_g_queue.append(len(goals))
    n_goal = np.argmax(ut.tally(n_g_queue))
    print "expected goals: ", n_goal

    # 2. collect queues for expected feature ids
    state_deque = [collections.deque(maxlen=queue_size) for _ in range(n_goal)]    
    feature_deque = [collections.deque(maxlen=queue_size) for _ in range(n_goal)]
    if enable_cstr:
        # a sequence of mu per partition
        feat_len = len(log['support_feature_values'][0])        
        cstr_deque    = [collections.deque(maxlen=queue_size) for _ in range(n_goal)]
        cstr_mu_deque = [[collections.deque(maxlen=queue_size) for _ in range(feat_len)] \
                         for _ in range(n_goal)]
        
    for i, goals in enumerate(log['goals'][-queue_size:]):
        for j, goal in enumerate(goals):
            if j>= n_goal: continue
            state_deque[j].append( goal[0] )
            feature_deque[j].append( goal[2] )
            if enable_cstr:
                assert len(goal)>3, "no cstr info"
                cstr_deque[j].append( goal[-1] )
                c_mu = goal[-2]['mu']
                for k, mu in enumerate( c_mu ):
                    cstr_mu_deque[j][k].append( mu )

    # 3. compute expected feature and constraint indices
    goal_states = []
    goal_features = []
    cstr_ids = []
    cstr_mu  = []
    cstr_counts = []
    for i in range(n_goal):
        f_id = np.argmax( ut.tally(feature_deque[i]) )
        goal_features.append( f_id )

        # expected goal id from support states (idx_traj)
        g_id = idx_traj.index( np.argmax(ut.tally(state_deque[i])) )
        goal_states.append( g_id )

        # expected constraints
        if enable_cstr:
            cstr_list  = np.array(list(cstr_deque[i]))
            cstr_count = ut.tally(cstr_list[ np.where(np.array(state_deque[i])==idx_traj[g_id])[0] ])
            #cstr_count = ut.tally(cstr_deque[i])
            cstr_id = np.argmax( cstr_count )
            cstr_ids.append( cstr_id )
            cstr_counts.append( cstr_count )
            print "cstr_id: ", cstr_id
            
            mus = []
            for k in range(feat_len):
                f_max   = np.amax(cstr_mu_deque[i][k]); f_min = np.amin(cstr_mu_deque[i][k])
                n_bins  = 20
                hist, _ = np.histogram(cstr_mu_deque[i][k], bins=n_bins, range=(f_min, f_max))
                mus.append( np.argmax(hist)/float(n_bins)*float(f_max-f_min)+f_min  )
            cstr_mu.append(mus)

    if return_params:
        d = {'goal_states': goal_states,
             'cstr_counts': cstr_counts}
    else: d = {}

    if enable_cstr:            
        return goal_features, cstr_ids, cstr_mu, d
    else:
        return goal_features,\
          [None for _ in range(len(goal_features))],\
          [None for _ in range(len(goal_features))], d


def goal_action_weight(state_idx, action_idx, goal_state_idx, states, roadmap):
    """TODO: need to handle different distance metric"""
    return NotImplementedError
    
    ga = states[goal_state_idx] - states[state_idx]
    ga_norm = np.linalg.norm(ga)
    if ga_norm>0.: ga /= ga_norm

    a  = states[roadmap[state_idx][action_idx]] - states[state_idx]
    a_norm = np.linalg.norm(a)
    if a_norm>0: a /= a_norm

    if ga_norm==0 and a_norm==0:
        return 1.
    else:
        return ((np.dot(ga,a)+1.)/2.)**2 #1.5 #**1. #**2 #3
    
def goal_action_weights(state_ids, action_ids, goal_state_ids, states, roadmap):
    """TODO: need to handle different distance metric"""
    return NotImplementedError
    
    goal_actions = [] #obs x support_states
    for state_idx, action_idx in zip(state_ids, action_ids):
        tmp = []
        for s in goal_state_ids:
            tmp.append( goal_action_weight(state_idx, action_idx, s, states, roadmap) )
        goal_actions.append(tmp)
    
    return goal_actions
    
def init_irl_params(n_observations, n_goals, support_policy, support_states, support_feature_ids,\
                    support_feature_state_dict, observations):

    if type(support_policy[support_policy.keys()[0]]) is dict:
        support_policy_dict = support_policy
        support_policy      = support_policy_dict[-1]
    else:
        support_policy_dict = None
                    
    if False:
        # random
        z     = np.random.randint(n_goals, size=n_observations)
        goals = [sample_gc(support_states, support_policy, support_feature_ids=support_feature_ids,\
                           support_feature_state_dict=support_feature_state_dict,)
                               for _ in range(n_goals)]
    else:
        # order
        if n_observations%n_goals == 0:
            chunksize = n_observations/n_goals
        else:
            chunksize = n_observations/n_goals+1

        cnt      = 0
        z        = np.zeros(n_observations).astype(int)
        goals    = []
        cstr_idx = len(support_policy[support_policy.keys()[0]])-1
        
        for i in range(0, n_observations, chunksize):
            ids = slice(i, i+chunksize)
            z[ids] = cnt
            cnt += 1
            if i+chunksize-1 < n_observations:
                state = observations[i+chunksize-1].state
            else:
                state = observations[-1].state
            s_idx = support_states.index(state)
            f_id  = support_feature_ids[s_idx]
            goals.append([ state, support_policy[state][cstr_idx].astype(np.float), support_feature_ids[s_idx],
                           f_id, None, cstr_idx])
    return goals, z
    






