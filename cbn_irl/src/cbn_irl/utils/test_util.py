import sys, copy
import rospy
import numpy as np, scipy

from cbn_irl.utils import misc, gen_traj as gt
from cbn_irl.viz import viz as v
from cbn_irl.path_planning import probabilistic_road_map_pose_5d as prm
from cbn_irl.path_planning import dijkstra_planning as dp

def test_on_discretized_space(env, agent, error, irl_goals, goal_dist,
                              bnirl, viz=False, distFunc=None):
    """
    Run on the discretized space as training
    """

    states  = agent.states
    goal    = env.get_goal_state()
    roadmap = env.roadmap

    
    # only for the last movement ------------------------------
    state, time = env.reset(), 0
    if distFunc is None:
        s_idx       = np.argmin(np.linalg.norm(states-state, axis=-1))        
        g_idx       = np.argmin(np.linalg.norm(states-goal, axis=-1))
    else:
        s_idx       = np.argmin(prm.pose_distance(states,state))        
        g_idx       = np.argmin(prm.pose_distance(states,goal))
        

    if irl_goals[-1][0] != g_idx:
        rewards = agent.get_rewards()
        rewards[np.where(rewards>0)]=0.
        rewards[g_idx] = 1.
        agent.set_rewards(rewards)
        values, param_dict = agent.solve_mdp(error, return_params=True)
    #----------------------------------------------------------

    if viz: raw_input("Press Enter to continue...")
    # Visualization
    while True:
        if time==0:
            state, time = env.reset(), 0
            if distFunc is None:
                s_idx       = np.argmin(np.linalg.norm(states-state, axis=-1))        
                g_idx       = np.argmin(np.linalg.norm(states-goal, axis=-1))
            else:
                s_idx       = np.argmin(prm.pose_distance(states,state))        
                g_idx       = np.argmin(prm.pose_distance(states,goal))
            traj        = []
            change_point = [0]
            passed_z    = []

        traj.append(states[s_idx])
        n_actions = len(roadmap[s_idx])

        # find z_k and g_{z_k}
        if irl_goals[-1][0] != g_idx and len(passed_z)==len(irl_goals):
            g_k = [g_idx, param_dict['q'], -1]
        else:
            for i in range(len(irl_goals)):
                if not(i in passed_z):
                    z_k = i
                    break
            g_k = irl_goals[z_k]

        # take an action
        a_idx = bnirl.find_action(s_idx, g_k)#, alpha=irl_alphas[0])
        a     = env.get_action(states[s_idx], states[roadmap[s_idx][a_idx]])
        next_state, r, done, _ = env.step(a)
        if viz: env.render()


        time += 1
        if time > 150 or (len(passed_z)>=len(irl_goals) and done):
            if done: print "Done!"
            traj.append(next_state)
            traj  = np.array(traj)
            state = np.array(env.reset())
            ## time  = 0
            if viz: v.subgoal_marker_pub( traj, 'sub_goals', marker_id=700 )
            break #continue #break
        state = next_state
        ## if done:
        ##     env.close()
        ##     sys.exit()

        if distFunc is None:
            s_idx = np.argmin(np.linalg.norm(states-state, axis=-1))        
        else:
            s_idx = np.argmin(prm.pose_distance(states,state))                

        # determine if reached on a sub goal
        if (distFunc is not None and distFunc(states[g_k[0]], state)<goal_dist) or\
          ( distFunc is None and np.linalg.norm(states[g_k[0]]-state)<goal_dist):
            if not(z_k in passed_z):
                passed_z.append(z_k)
                change_point.append(time-1)
            print "passed z_id={} and z_ids={}, #goals={}".format(z_k, passed_z, len(irl_goals))
        if viz: rospy.sleep(0.2)
    change_point.append(len(traj)-1)

#    trajs = []
#    for i in range(len(change_point)-1):
#        trajs.append( traj[change_point[i]:change_point[i+1]+1] )

        ## v.subgoal_marker_pub( trajs[-1], 'partition_{}'.format(i), marker_id=1000*(i+1) )

    ## sub_goals = []
    ## for i, traj in enumerate(trajs):
    ##     sub_goals.append(traj[-1])
    ## v.subgoal_marker_pub( sub_goals, 'sub_goals_mesh', marker_id=0,
    ##                       mesh_resource=env.mesh_resource['clamp_4'])

    ## sub_cstrs = []
    ## for i, traj in enumerate(trajs):
    ##     if len(irl_goals)-1<i or irl_goals[i][-1]>0: continue
    ##     sub_cstrs+=list(traj)
    ## v.subgoal_marker_pub( sub_cstrs, 'sub_constraints', marker_id=0,)
        
    ## from IPython import embed; embed()#; sys.exit()                

    return traj, done, {}


def get_dijkstra_trajs(env, agent, irl_goals):
    states  = agent.states
    ## goal    = env.get_goal_state()
    roadmap = env.roadmap
    
    state, time = env.reset(), 0
    
    trajs = []
    for i, irl_goal in enumerate(irl_goals):
        s_idx       = np.argmin(prm.pose_distance(states, state))        
        g_idx       = irl_goal[0]

        start = states[s_idx]
        goal  = states[g_idx]
        
        new_traj, rids = dp.dijkstra_planning(env, start, goal, roadmap,
                                              states, distFunc=prm.pose_distance,
                                              verbose=False)
        trajs.append(new_traj)
        if len(new_traj)>0:
            state = new_traj[-1]

    return trajs
    


def test_on_continuous_space(env, irl_goals, bnirl, env_xml, manip_name,
                             feature_fn,
                             cstr_feat_id=None, cstr_ths=None,
                             use_discrete_state=True,
                             default_trajs=None, return_traj=False,
                             use_avg_filter=True, 
                             viz=False):
    """
    Run on the continuous space using trajopt
    """
    from cbn_irl.path_planning import opt_planner as op
    
    states  = env.states
    goal    = env.get_goal_state()


    state, time = env.reset(), 0
    start = copy.copy(state)
    traj     = []
    traj_ext = []
    s        = misc.list_quat2list_rpy(start)    

    # Remove same goals
    new_irl_goals = []
    new_default_trajs = []
    for i, irl_goal in enumerate(irl_goals):
        if i>0 and irl_goals[i][0]==irl_goals[i-1][0]: continue

        new_irl_goals.append(irl_goal)
        if default_trajs is not None and len(default_trajs)>i:
            print "interpolate the default trajectory"
            l = len(default_trajs[i])
            if int(l*0.3)<3:
                new_default_trajs.append(default_trajs[i])
            else:
                times     = np.linspace(0., 1., l  )
                new_times = np.linspace(0., 1., int(l*0.3) )    
                interp_traj = gt.interpolationData(times, default_trajs[i], new_times, enable_spline=True)
                new_default_trajs.append(interp_traj)
        elif default_trajs is None:
            continue
        else:
            new_default_trajs.append(default_trajs[i])
    default_trajs = new_default_trajs if len(new_default_trajs)>0 else None
    irl_goals = new_irl_goals

    if np.shape(states)[-1]>7:
        enable_extra_state = True
    else:
        enable_extra_state = False
        
    # TEMP
    ## viz_info(env, default_trajs, irl_goals)

    for i, irl_goal in enumerate(irl_goals):
        
        if use_discrete_state:
            g     = misc.list_quat2list_rpy(states[irl_goal[0]])
            if enable_extra_state: g_ext = states[irl_goal[0]][7:]
        else:
            g = misc.list_quat2list_rpy(irl_goal[0])
            if enable_extra_state: g_ext = irl_goal[0][7:]

        ineq_fn = None
        cost_fn = None
        if cstr_feat_id is not None and irl_goal[-1]==0:
            ineq_fn = bnirl.make_ineq_cstr_fn(feature_fn, env, np.array(irl_goal[3]),
                                              cstr_feat_id, cstr_ths, scale=100.)
            if ineq_fn(g) > 0:
                print ("Adjust an invalid goal")
                g = search_valid_state(misc.list_rpy2list_quat(g), ineq_fn, env)
                g = misc.list_quat2list_rpy(g)

            ineq_fn = bnirl.make_ineq_cstr_fns(feature_fn, env, np.array(irl_goal[3]),
                                               cstr_feat_id, cstr_ths,
                                               scale=1e+6)
            ## cost_fn = bnirl.make_cost_fn(feature_fn, env, np.array(irl_goal[3]),
            ##                              cstr_feat_id, scale=1.,)
                
        # adjust invalid goals
        if cstr_feat_id is not None and len(irl_goals) > i+1 and irl_goals[i+1][-1]==0 and False:
            next_ineq_fn = bnirl.make_ineq_cstr_fn(feature_fn, env,
                                                   np.array(irl_goals[i+1][3]),
                                                   cstr_feat_id, cstr_ths)
            if next_ineq_fn(g) > 0:
                rospy.logerr("Adjust an invalid goal wrt the next ineq")
                g = search_valid_state(misc.list_rpy2list_quat(g), next_ineq_fn, env)
                g = misc.list_quat2list_rpy(g)
                if next_ineq_fn(g)>0:
                    rospy.logerr("A sub-goal does not satisfy constraints. The goal adjustment also did not work.")

        # adjust euler angles
        if abs(s[3]-g[3])>np.pi:
            if g[3]-s[3] > 0: g[3]-=np.pi*2.
            else:             g[3]+=np.pi*2.   
        if abs(s[4]-g[4])>np.pi:
            if g[4]-s[4] > 0: g[4]-=np.pi*2.
            else:             g[4]+=np.pi*2.   
        if abs(s[5]-g[5])>np.pi:
            if g[5]-s[5] > 0: g[5]-=np.pi*2.
            else:             g[5]+=np.pi*2.   
                
                    
        ## from mdp import reward_baxter
        ## c_fn = reward_baxter.make_dist_cost(prm.rpy_pose_distance, g) 

        ref_traj = None
        if default_trajs is not None and len(default_trajs) > i and \
          len(default_trajs[i])>=2 and \
            prm.pose_distance(misc.list_rpy2list_quat(g), default_trajs[i][-1])<0.05:

            ref_traj = default_trajs[i]
            #ref_traj = [misc.list_rpy2list_quat(s)] + list(ref_traj) + [misc.list_rpy2list_quat(g)]
            
            delta = np.amax(ref_traj, axis=0)-np.amin(ref_traj, axis=0)
            l = 0
            for j in range(3):
                l = max(l, np.abs(delta[j])/env.action_space.high[j])
            if len(ref_traj) < l:
                print "Reinterpolate default_traj from {} to {}".format(len(ref_traj), int(l*1.5) )
                ref_traj = gt.get_interpolation_data(np.array(ref_traj), len_traj=int(l*1.5) )
          
            ref_rpy_traj = []
            for j in range(len(ref_traj)):
                ref_rpy_traj.append( misc.list_quat2list_rpy(ref_traj[j]) )
            ref_traj = [s]+list(ref_rpy_traj)+[g]        
            
        last_traj_size = len(traj)
        ## traj += list(op.plan(env_xml, manip_name, s, g, traj=copy.copy(ref_traj),
        ##                      ineq_fn=ineq_fn))        
        temp = op.plan(env_xml, manip_name, s, g, traj=copy.copy(ref_traj),
                            ineq_fn=ineq_fn)#, c_fn=c_fn)


        if type(temp) is not list: temp = temp.tolist()
        traj += temp

        if enable_extra_state:
            traj_ext += np.tile(g_ext, (len(temp), 1) ).tolist()

        ## if irl_goal[-1]==0:
            ## v.feature_over_traj(env, ref_traj, feature_fn)
            ## v.feature_over_traj(env, mvg_filter(ref_traj), feature_fn)
            ## v.feature_over_traj(env, temp, feature_fn)
            ## print np.sum(np.abs(ref_traj-np.array(temp)))
                
        if len(traj)-last_traj_size == 0 and prm.pose_distance(misc.list_rpy2list_quat(traj[-1]), misc.list_rpy2list_quat(g))>0.02:
            print ("No progress and there is a jump")
            from IPython import embed; embed(); sys.exit()                
            return None, False

        s = g


    # final reaching motion
    if (use_discrete_state and prm.pose_distance(states[irl_goals[-1][0]], goal)>0.005) or (use_discrete_state is False and prm.pose_distance(irl_goals[-1][0], goal)>0.005):
        g = misc.list_quat2list_rpy(goal)            

        temp = list(op.plan(env_xml, manip_name, s, g))
        traj += temp
        if enable_extra_state:
            traj_ext += np.tile(goal[7:], (len(temp), 1) ).tolist()

    new_traj = []
    pose_traj = []
    for i, s in enumerate(traj):
        if enable_extra_state is False:
            new_traj.append(misc.list_rpy2list_quat(s))
        else:
            new_traj.append(misc.list_rpy2list_quat(s)+traj_ext[i])
        pose_traj.append(misc.list2Pose(new_traj[-1]))

    if viz: v.pose_array_pub(pose_traj, "predicted_pose_array", "base")
    if use_avg_filter:
        ## new_traj = mvg_filter(new_traj)
        l = len(new_traj)
        times     = np.linspace(0., 1., l  )

        delta = np.abs(np.array(new_traj)[1:]-np.array(new_traj)[:-1])
        delta = np.amax(delta, axis=0)
        mult = 1
        for j in range(3):
            mult = max(mult, np.abs(delta[j])/env.action_space.high[j] + 1 )                    
        new_times = np.linspace(0., 1., l*int(mult)*2+40 )    
        new_traj = gt.interpolationData(times, new_traj, new_times, enable_spline=False)            


    if viz: v.subgoal_marker_pub( new_traj, 'predicted_traj', marker_id=700 )
    if return_traj: return new_traj
        
    ## raw_input("Press Enter to continue...")
    done = False
    n_collisions = 0
    for t, s in enumerate(new_traj):
        if t==0: continue

        a = env.get_action(state, s) #pos+rpy
        state, r, done, info = env.step(a)

        if info.get('state_validity', True) is False:
            print t
            n_collisions += 1
        ## elif info['action_validity'] is False:
        ##     from IPython import embed; embed(); sys.exit()            
        
        if done:
            print "Done!"
            ## v.subgoal_marker_pub( traj, 'predicted_traj', marker_id=700 )
            break
        if viz: rospy.sleep(0.05)

    info['n_collisions'] = n_collisions
    return new_traj, done, info





def search_valid_state(g, next_ineq_fn, env):
    print ("Search a valid state that satisfies constraints")

    def make_objective_fn(x):
        def fn(state):
            return prm.pose_distance(x, state)*10000.
            ## return np.linalg.norm(x-state) 
        return fn

    fn = make_objective_fn(g)

    # f(x) >= 0
    cons = []
    ## cons.append({'type': 'ineq', 'fun': lambda x: x[0]-env.observation_space.low[0] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: x[1]-env.observation_space.low[1] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: x[2]-env.observation_space.low[2] })

    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[0] - x[0] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[1] - x[1] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.high[2] - x[2] })

    cons.append({'type': 'ineq', 'fun': lambda x: -next_ineq_fn(x)*10000. })
    #cons.append({'type': 'ineq', 'fun': lambda x: -next_ineq_fn(list(x)+list(g[3:])) })
  
    ## # f(x) <= 0
    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.low[0]-x[0] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.low[1]-x[1] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: env.observation_space.low[2]-x[2] })

    ## cons.append({'type': 'ineq', 'fun': lambda x: x[0]-env.observation_space.high[0] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: x[1]-env.observation_space.high[1] })
    ## cons.append({'type': 'ineq', 'fun': lambda x: x[2]-env.observation_space.high[2] })

    #cons.append({'type': 'ineq', 'fun': lambda x: next_ineq_fn(x) })

    bounds = ((env.observation_space.low[0], env.observation_space.high[0]),
              (env.observation_space.low[1], env.observation_space.high[1]),
              (env.observation_space.low[2], env.observation_space.high[2]),
              (env.observation_space.low[3], env.observation_space.high[3]),
              (env.observation_space.low[4], env.observation_space.high[4]),
              (env.observation_space.low[5], env.observation_space.high[5]),
              (env.observation_space.low[6], env.observation_space.high[6]),
              )
    
    res = scipy.optimize.minimize(fn, g, method='SLSQP', bounds=bounds,
                                  constraints=tuple(cons),
                                  options={'maxiter': 5000,})
    print res

    return list(res.x)




def viz_info(env, trajs, irl_goals):

    sub_goals = []
    for i, traj in enumerate(trajs):
        sub_goals.append(traj[-1])
    v.subgoal_marker_pub( sub_goals, 'sub_goals_mesh', marker_id=0,
                          mesh_resource=env.mesh_resource['clamp_4'])

    sub_cstrs = []
    for i, traj in enumerate(trajs):
        if len(irl_goals)-1<i or irl_goals[i][-1]>0: continue
        sub_cstrs+=list(traj)
    v.subgoal_marker_pub( sub_cstrs, 'sub_constraints', marker_id=0,)
    

def mvg_filter(x):
    """Moving average filter
    @x input trajectory (LENGTH, DIMENSION)
    """
    if type(x) is list:
        x = np.array(x)

    filter_size = 5
    
    # moving average filter and zero velocity padding
    new_x = []
    for i in xrange(len(x[0])):
        d = [x[0,i]]*filter_size + list(x[:,i]) + [x[-1,i]]*filter_size*2
        d = np.convolve(d,np.ones((filter_size,))/float(filter_size), mode='valid')
        new_x.append(d)
    new_x = np.array(new_x).T

    return new_x
