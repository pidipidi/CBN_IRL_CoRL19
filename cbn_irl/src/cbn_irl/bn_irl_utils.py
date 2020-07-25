from random import random
import numpy as np
from scipy.stats import dirichlet, trim_mean
import copy

class Clusters:
    def __init__(self, N, K, assignments, G, Z, ids):
        self.N = N #Integer: partitions
        self.K = K #Array:   occurences per partitions
        self.assignments = assignments #Array: 
        self.G = G #Vector{Vector{Goal}}
        self.Z = Z #Array
        self.ids = ids #Array


"""
Given an array, finds the number of times the array elements occur
"""
def tally(zd):
    ret = np.zeros(np.amax(zd).astype(int)+1)
    for k in zd:
        ret[k] += 1
    return ret

                        
"""
Given a vector of assignements, calculates the CRP probability vector,
setting the last element as the probability of instantiating a new cluster
"""
def CRP(assignments, alpha=1., use_clusters=True):

    occurences = tally(assignments)
    _sum       = float(sum(occurences))

    if use_clusters:
        denom = _sum-1.+alpha
        prob_size = len(occurences)+1
    else:
        denom = _sum-1.
        prob_size = len(occurences)

    prob_vector = np.zeros(prob_size)
    for i in range(len(occurences)):
        prob_vector[i] = float(occurences[i]) / denom

    if use_clusters: prob_vector[-1] = alpha / denom

    return prob_vector


"""
Given a list of trajectories, get unique observations
"""
def trajs2observations(trajs, idx_trajs, roadmap, states):
    import bn_irl_common as bic
    observations = []
    indices      = []
    length = len(trajs[0])

    for i in range(len(trajs[0])):            
        for traj, idx_traj in zip(trajs, idx_trajs):
            ## for i in range(len(traj)):            
            idx = idx_traj[i]

            if i < len(traj)-1:
                action_idx = None
                for j, next_idx in enumerate(roadmap[idx]):
                    if next_idx == idx_traj[i+1]:
                        action_idx = j
                        break
                if action_idx is None:
                    continue

                unique=True
                for obs in observations:
                    if obs.state==idx and obs.action==action_idx:
                        unique=False
                if unique is False: continue

                observations.append(bic.O(idx, action_idx))
                
            else:
                unique=True
                for obs in observations:
                    if obs.state==idx:
                        unique=False
                if unique is False: continue

                action_idx = roadmap[idx].index(idx)
                observations.append(bic.O(idx, action_idx))

    ## import matplotlib.cm as cm
    ## import matplotlib.pyplot as plt
    ## fig = plt.figure()
    ## for traj in trajs:
    ##     plt.plot(np.array(traj)[:,0], np.array(traj)[:,1], 'r-')
    ## tmp = np.array(tmp)
    ## plt.scatter(tmp[:,0], tmp[:,1], c='r')
    ## plt.show()
    ## sys.exit()
    return observations


"""
Given a list of index trajectories, get partition numbers
"""
def traj2partition(idx_traj, roadmap, states, z, observations):
    import bn_irl_common as bic

    z_traj = []
    for i in range(len(idx_traj)):            
        ## for i in range(len(traj)):            
        idx = idx_traj[i]

        if i < len(idx_traj)-1:
            action_idx = None
            for j, next_idx in enumerate(roadmap[idx]):
                if next_idx == idx_traj[i+1]:
                    action_idx = j
                    break
        else:
            action_idx = roadmap[idx].index(idx)

        for j, obs in enumerate(observations):
            if obs.state==idx and obs.action==action_idx:
                break
            
        if z is not None:
            z_traj.append(z[j])
        else:
            z_traj.append(0)
            
    return z_traj


"""
Given a list of state trajectories, get a list feature trajectories
"""
def trajs2featTrajs(idx_trajs, feat_map, action_enabled=False):
    # features / partitions
    feat_trajs = []
    for idx_traj in idx_trajs:
        ## fs = []
        ## for i, s in enumerate(idx_traj):
        ##     if action_enabled and False:
        ##         return NotImplementedError
        ##         ## if i == 0: a = 0
        ##         ## else:      a = s[i]-s[i-1]
        ##         ## fs.append( feature_fn(s, action=a) )
        ##     else:
        ##         fs.append( feat_map[s] )
        ##         ## fs.append( feature_fn(s) )
        fs = feat_map[idx_traj]
        feat_trajs.append(fs) #set to the first partition
    return feat_trajs
    


"""
"""
def animation_goals_2d(env, trajs, log, states, n_steps=10, enable_cstr=False, queue_size=1000,
                       save_clip=False):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.lines import Line2D
    import collections

    # setup
    goal_log, z_log, observations_log = log['goals'], log['z'], log['observations']
    play_offset = len(goal_log)/50    

    # maximum number of partitions
    nz_max = 0
    for g in goal_log:
        if nz_max<len(g):
            nz_max = len(g)

    # partition image
    N = 60
    X, Y = np.mgrid[env.observation_space.low[0]:env.observation_space.high[0]:complex(0, N),
                    env.observation_space.low[1]:env.observation_space.high[1]:complex(0, N)]
    
    # matplot setup
    fig = plt.figure(figsize=(12,6))
    # Figure 1 --------------------------------------------------------------
    ax1 = fig.add_subplot(121) 
    # environments
    trajs = np.array(trajs)
    for traj in trajs:
        ax1.plot(traj[:,0],traj[:,1], 'r-', alpha=0.2)
    ax1.plot(env.objects[:,0], env.objects[:,1], 'ko')


    colors = cm.rainbow(np.linspace(0, 1, nz_max))
    z_lns1 = []
    for i in range(nz_max):
        z_ln1, = ax1.plot([], [], c=colors[i], marker='o', markersize=7., linestyle='', alpha=0.5,
                        markeredgecolor='none')
        z_lns1.append(z_ln1)
        ## plt.scatter(obs[:,0], obs[:,1], c=colors[i], alpha=0.2)        
        ## arrw = np.array([states[roadmap[o.state][o.action]]-states[o.state]
        ##                  for o in obs_zi])
        #plt.quiver(obs[:,0], obs[:,1], arrw[:,0], arrw[:,1], colors[i], scale=100.)

    ax1.set_xlim( [env.observation_space.low[0], env.observation_space.high[0]])
    ax1.set_ylim( [env.observation_space.low[1], env.observation_space.high[1]])
    ## ax1.set_xlabel('X Label')
    ## ax1.set_ylabel('Y Label')
    plt.xticks([]),plt.yticks([])

    lines  = [#Line2D([0], [0], color='r', linewidth=3, linestyle='-'),
              plt.scatter(np.linspace(-10,-5,len(colors)), np.linspace(-10,-5,len(colors)),\
                       marker='o', c='g', s=50, alpha=0.2)]
    labels = [#'Demo.',
              'Partition'] 
    plt.legend(lines, labels, fontsize='large', loc=1)

    # Figure 2 --------------------------------------------------------------
    ax2 = fig.add_subplot(122)
    ax2.plot(env.objects[:,0], env.objects[:,1], 'ko')

    # sub-goals
    alphas = np.linspace(0.2, 1.0, n_steps)[::-1]
    g_lns = []
    for i in range(n_steps):
        g_ln, = ax2.plot([], [], c='g', marker='P', markersize=15., linestyle='', alpha=alphas[i])
                        ## markeredgecolor='none')
        g_lns.append(g_ln)
    explored_g_ln, = ax2.plot([], [], c='k', marker='o', markersize=5., linestyle='', alpha=0.2)

    z_ln2, = ax2.plot([], [], c='r', marker='o', markersize=5., linestyle='', alpha=0.3,
                      markeredgecolor='none')

    ax2.set_xlim( [env.observation_space.low[0], env.observation_space.high[0]])
    ax2.set_ylim( [env.observation_space.low[1], env.observation_space.high[1]])
    ## ax2.set_xlabel('X Label')
    ## ax2.set_ylabel('Y Label')
    plt.xticks([]),plt.yticks([])

    lines  = [plt.scatter(np.linspace(-10,-5,len(colors)), np.linspace(-10,-5,len(colors)),\
                          marker='P', c='g', s=50, alpha=1.),
              plt.scatter(np.linspace(-10,-5,len(colors)), np.linspace(-10,-5,len(colors)),\
                       marker='o', c='r', s=50, alpha=0.2)]
    labels = ['Subgoal', 'State w/ \nConstraints'] 
    plt.legend(lines, labels, fontsize='large', loc=3)
    
    #-------------------------------------------------------------------------
    # Data structure
    #-------------------------------------------------------------------------
    
    data_goal = []
    data_cstr = []
    data_z    = []
    data_goal_ids = [ ]
    data_cnt  = []

    nz = len(goal_log[-1])
    seq_goals = []
    seq_cstrs = []
    seq_zs    = [] #len(z_log[-1])
    ## from IPython import embed; embed()#; sys.exit()


    # take a sequence of goal and partitions
    for i, (goals, zs) in enumerate(zip(goal_log, z_log)):
        # sub-goals #TODO: varying number of goals
        g = [] 
        for j, goal in enumerate(goals):
            g.append( log['support_states'].index(goal[0]) )
        seq_goals.append(g)

        # partitions                
        seq_zs.append(zs)

        if enable_cstr:
            # sub constraints        
            c = []
            for j, goal in enumerate(goals):
                if len(goal)>3:
                    c.append( goal[-1] )
                else:
                    c.append(-1)
            seq_cstrs.append(c)       
    seq_goals = np.array(seq_goals) # support state indices (0~)
        

    # average a block of sequence
    data_goal = []
    data_z    = []    
    n_goal_deque = collections.deque(maxlen=queue_size)
    goal_deque = [collections.deque(maxlen=queue_size) for _ in range(nz_max)]
    cstr_deque = [collections.deque(maxlen=queue_size) for _ in range(nz_max)]
    for i, (goals, zs) in enumerate(zip(seq_goals, seq_zs)):       
        # goal
        for j, goal in enumerate(goals):
            goal_deque[j].append(goal)  #support_state index
        n_goal_deque.append(len(goals))

        if enable_cstr:
            for j, seq_cstr in enumerate(seq_cstrs[i]):
                cstr_deque[j].append( seq_cstr ) 

        # -------------------------------------------------
        if i>0 and not(i%play_offset==0) and i!=len(seq_goals)-1: continue
        data_cnt.append(i)
        
        expected_n_goal = np.argmax(tally(n_goal_deque))

        # expected goal
        g = []
        goal_in_support_state_ids = []
        for j in range(expected_n_goal):
            idx = np.argmax(tally(goal_deque[j])) # support state index
            g.append( states[log['support_states'][idx]] )
            goal_in_support_state_ids.append(idx)
        data_goal.append(g)


        # partition (from current sample)
        Z = {j: [] for j in range(len(tally(zs)))}
        for j in range(nz):
            ids = [k for k, v in enumerate(zs) if v==j]
            obs_zi = np.array(observations_log)[ids]
            #if len(obs_zi)==0: continue
            Z[j] = np.array([states[o.state] for o in obs_zi])
        data_z.append(Z)
        
        Z = {j: [] for j in range(len(goal_in_support_state_ids))}
        for j in range(len(goal_in_support_state_ids)):
            if j==0:
                start = 0
            else:
                start = goal_in_support_state_ids[j-1]
            end   = goal_in_support_state_ids[j]+1            
            Z[j] = np.array([ states[s] for s in log['support_states'][start:end] ])


        # sub constraints
        if enable_cstr:
            C = None
            cstr_ids = []
            for j in range(expected_n_goal):

                g_id       = goal_in_support_state_ids[j]

                cstr_list  = np.array(list(cstr_deque[j]))
                cstr_count = tally(cstr_list[ np.where(goal_deque[j]==g_id)[0] ])
                cstr_id    = np.argmax( cstr_count )
                cstr_ids.append( cstr_id )
                if cstr_id > 0: continue  #temp

                try:
                    if C is None: C = Z[j]
                    else:         C = np.vstack([C, Z[j]])
                except:
                    print "not enough partitions given higher number of expected goals:\
                    {} / {}".format(None, expected_n_goal)
            data_cstr.append(C)


    # attach tails
    for _ in range(n_steps):
        data_cnt.append(data_cnt[-1])
        data_goal.append( data_goal[-1] )
        data_z.append(data_z[-1])
        if enable_cstr:
            data_cstr.append( data_cstr[-1] )

            
    #Visualize goal and partitions
    def update_goals(i):
        # partitions
        for j in range(nz_max):
            if j < len(data_z[i].keys()) and len(data_z[i][j])>0:
                z_lns1[j].set_data(data_z[i][j][:,0], data_z[i][j][:,1])
            else:
                z_lns1[j].set_data([], [])                            
                
        # sub constraints
        if enable_cstr and data_cstr[i] is not None:
            try:
                z_ln2.set_data(data_cstr[i][:,0], data_cstr[i][:,1])
            except:
                z_ln2.set_data([],[])
                ## from IPython import embed; embed(); sys.exit()                
            ## z_ln2.set_data(data_cstr[i].T)
        else:
            z_ln2.set_data([],[])
            

        # exploration
        ## explored_g_ln.set_data(data_goal[i][:,0], data_goal[i][:,1])
        
        # sub-goals
        for ln_num in range(n_steps):
            if i-ln_num<0: continue
            goal = np.array(data_goal[i-ln_num])
            ## if len(np.shape(goal))==1: goal=goal[np.newaxis,:]
            g_lns[ln_num].set_data(goal[:,0], goal[:,1])
        
        ax1.set_title("Sampled {} Partitions at {} steps".format(len(data_z[i]), data_cnt[i]))
        ax2.set_title("Expectation of Sub-goals and Constraints".format( len(data_goal[i]) ))
        fig.suptitle("{} steps".format(data_cnt[i]), fontsize=16)


    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_goals, frames=len(data_goal), #, fargs=([data_goal, data_z]),
                                  interval=1, blit=False, repeat=False)
    if save_clip:
        ani.save('ani.mp4', writer='ffmpeg', fps=5)
    plt.show()
        

def animation_goals_3d(pose_trajs, log, poses, min_pos=None, max_pos=None):
    """
    pose_trajs:
    poses:
    """
    if type(pose_trajs) is list: pose_trajs = np.array(pose_trajs)
    
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d.axes3d as p3
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if min_pos is not None or max_pos is not None:
        ax.set_xlim( [min_pos[0], max_pos[0]])
        ax.set_ylim( [min_pos[1], max_pos[1]])
        ax.set_zlim( [min_pos[2], max_pos[2]] )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.plot(pose_trajs[:,0],pose_trajs[:,1],pose_trajs[:,2], 'r-', alpha=0.5)
    ln, = ax.plot([], [], [], c='m', marker='*', markersize=10., linestyle='')

    # setup
    goal_log, z_log, observations_log = log['goals'], log['z'], log['observations']


    def update_goals(i, goals):
        ln.set_data(goals[i][:,0], goals[i][:,1])
        ln.set_3d_properties(goals[i][:,2])
        plt.title("Timestep: {}, Groups: {}".format(i, len(goals[i])))
                    
    # init
    ## sub_goals = []
    ## for goal in goal_log[0]:
    ##     sub_goals.append(poses[goal[0]][:3])
    ## sub_goals = np.array(sub_goals)
    ## ax.plot(sub_goals[:,0], sub_goals[:,1], sub_goals[:,2], 'm*', markersize=12)

    ## from IPython import embed; embed(); sys.exit()

    data = []
    for i, (goals, zs) in enumerate(zip(goal_log, z_log)):
        if i%100 != 0 and i != len(goal_log)-1: continue
        sub_goals = []
        for goal in goals:
            ## from IPython import embed; embed(); sys.exit()                            
            sub_goals.append(poses[goal[0]][:3])

        sub_goals = np.array(sub_goals)
        data.append(sub_goals)
        
    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_goals, frames=len(goal_log), fargs=([data]),
                                  interval=50, blit=False, repeat=False)

    plt.show()


def viz_convergence(states, idx_trajs, log, queue_size=1000, cstr_enabled=False):
    """Plot the expected goal distributions
    """
    import collections
    import bn_irl_common as bic
    import matplotlib.cm as cm

    new_idx_traj = []
    for idx in idx_trajs[0]:
        if not(idx in new_idx_traj):
            new_idx_traj.append(idx)
    idx_traj=new_idx_traj

    # expected goals and constraints from a demonstration
    goal_features, cstr_ids, cstr_mus, param_dict = bic.get_expected_goal(log, states,
                                                                          enable_cstr=cstr_enabled,
                                                                          queue_size=queue_size,
                                                                          idx_traj=idx_traj,
                                                                          return_params=True)
    goal_states     = param_dict.get('goal_states', [])
    cstr_counts     = param_dict.get('cstr_counts', [])
    expected_n_goal = len(goal_states)

    goal_counts = [np.zeros( len(idx_traj) ) for _ in range(expected_n_goal)]
    for idx, (goal, z) in enumerate(zip(log['goals'], log['z'])):
        for j in range(expected_n_goal):
            if j<len(goal):
                i = idx_traj.index(goal[j][0])
                goal_counts[j][i] += 1
                
    import matplotlib.pyplot as plt
    fig = plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, expected_n_goal))
    for i in range(expected_n_goal):
        if i>9: continue
        ax = fig.add_subplot(expected_n_goal, 2, i*2+1)
        plt.bar(range(len(goal_counts[i])), goal_counts[i])
        if cstr_enabled:
            ax = fig.add_subplot(expected_n_goal, 2, i*2+1+1)
            plt.bar(range(len(cstr_counts[i])), cstr_counts[i])
            ## plt.plot(exp_cstr_ids[i], color=colors[i])
    
    ## ax4 = fig.add_subplot(614)
    ## plt.plot(gs_x_std)
    ## ax5 = fig.add_subplot(615)
    ## plt.plot(expected_n_goal)
    ## plt.plot(losses)
    plt.show()
    



def visualization(env, z, goals, observations, states, support_states, support_feature_state_dict,
                  trajs=None, alpha=1., punishment=5., q_mat=None):

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

    occurrence = tally(z)
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

        #goal_states = bic.get_state_goal_from_feature(goals[i][2], support_states, support_feature_ids)
        goal_states = support_feature_state_dict[goals[i][2]]           
        
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
