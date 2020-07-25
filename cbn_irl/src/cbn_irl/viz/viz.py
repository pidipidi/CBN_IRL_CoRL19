import sys
import numpy as np
import PyKDL
import matplotlib.pyplot as plt
import matplotlib.cm as cm

############################################# 2D ######################################
def reward_plot(rewards, states):

    plt.scatter(states[:,0], states[:,1], c=np.squeeze(rewards), s=35,
                vmin=np.amin(rewards), vmax=np.amax(rewards), edgecolors='none')
    plt.colorbar()
    plt.show()


def reward_value_plot(rewards, values, states, trajs=None, save_plot=False, filename='test.png'):
    if type(states) is list: states = np.array(states)
        
    fig = plt.figure(figsize=(12,4))
    
    fig.add_subplot(1,2,1)
    plt.scatter(states[:,0], states[:,1], c=rewards, s=25,
                vmin=np.amin(rewards), vmax=np.amax(rewards), edgecolors='none')
    plt.colorbar()

    fig.add_subplot(1,2,2)
    plt.scatter(states[:,0], states[:,1], c=values, s=25,
                vmin=np.amin(values), vmax=np.amax(values), edgecolors='none')
    plt.colorbar()

    if trajs is not None:
        if type(trajs) is list: trajs = np.array(trajs)
        for traj in trajs:
            plt.plot(traj[:,0], traj[:,1], 'r-')    

    if save_plot:
        fig.savefig(filename, format='png')
    else:
        plt.show()


def traj_plot(demo_traj, fitted_traj, objects, xlim, ylim):
    if type(demo_traj) is list: demo_traj = np.array(demo_traj)
    if len(demo_traj.shape)==2:
        demo_traj = np.expand_dims(demo_traj, axis=0)

    plt.plot(objects[:,0], objects[:,1], 'ko')
    for traj in demo_traj:
        plt.plot(traj[:,0], traj[:,1], 'r-')

    if fitted_traj is not None:
        if type(fitted_traj) is list: fitted_traj = np.array(fitted_traj)
        if len(fitted_traj.shape)==2:
            fitted_traj = np.expand_dim(fitted_traj, axis=0)
        for traj in fitted_traj:
            plt.plot(traj[:,0], traj[:,1], 'b-')
            
    ## plt.plot(self.start_state[0], self.start_state[1], 'rx')
    ## plt.plot(self.goal_state[0], self.goal_state[1], 'r^')        
    plt.xlim( xlim )
    plt.ylim( ylim )
    plt.show()
    

def subgoals_plot(subgoals, objects, states, env, traj=None):

    fig = plt.figure(figsize=(8,8))
    
    colors = cm.rainbow(np.linspace(0, 1, len(subgoals)))
    states = [states[subgoals[i][0]] for i in range(len(subgoals)) ]

    for i, s in enumerate(states):
        plt.scatter(s[0], s[1], c=colors[i], s=200, edgecolors='none')
        #plt.text(s[0], s[1]-5, str(i) )

    # start-goal
    plt.scatter(env.goal_state[0], env.goal_state[1], marker='*', c='r', s=400)
    plt.scatter(env.start_state[0], env.start_state[1], marker='*', c='r', s=400)

    #objects
    plt.plot(objects[:,0], objects[:,1], 'ko')

    # traj
    if traj is not None:
        if type(traj) is list: traj = np.array(traj)
        print np.shape(traj)
        plt.plot(traj[:,0], traj[:,1], 'r-')
        
    
    plt.xlim( [env.observation_space.low[0], env.observation_space.high[0] ])
    plt.ylim( [env.observation_space.low[1], env.observation_space.high[1] ])
    plt.axis('off')    
    plt.show()

    fig.savefig("temp.png", format='png')
    fig.savefig("temp.pdf", format='pdf')


############################################# 3D ######################################
def reward_plot_3d(rewards, states, env=None, demo_traj=None):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    states  = states[rewards>0]
    rewards = rewards[rewards>0]
    
    ax.scatter(states[:,0], states[:,1], states[:,2],
               c=rewards, #np.squeeze(rewards).tolist(),
               marker='o', edgecolors='none', s=35,
               vmin=np.amin(rewards), vmax=np.amax(rewards), alpha=0.1)
    
    if env is not None:
        ax.scatter(env.start_state[0], env.start_state[1], env.start_state[2],
                   c='b', marker='^',edgecolors='none', s=505 )
        ax.scatter(env.goal_state[0], env.goal_state[1], env.goal_state[2],
                   c='r', marker='^',edgecolors='none', s=505 )
        for key in env.object_dict:
            if key.find('l_palm')>=0: continue
            ax.scatter(env.object_dict[key][0], env.object_dict[key][1], env.object_dict[key][2],
                       c='g', marker='*',edgecolors='none', s=505 )

    if demo_traj is not None:
        if type(demo_traj) is list: demo_traj = np.array(demo_traj)
        if len(demo_traj.shape)==2:
            demo_traj = np.expand_dims(demo_traj, axis=0)
        for traj in demo_traj:
            plt.plot(traj[:,0], traj[:,1], traj[:,2], 'r-')
        
    #plt.colorbar()
    if env is not None and False:
        ax.set_xlim( [env.observation_space.low[0], env.observation_space.high[0] ])
        ax.set_ylim( [env.observation_space.low[1], env.observation_space.high[1] ])
        ax.set_zlim( [env.observation_space.low[2], env.observation_space.high[2] ])
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
        
    plt.show()


def reward_value_3d(rewards, values, states, trajs=None, save_plot=False, filename='test.png', env=None):
    if type(states) is list: states = np.array(states)
        
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(states[:,0], states[:,1], states[:,2], c=np.squeeze(rewards),
               marker='o', edgecolors='none', s=35,
               vmin=np.amin(rewards), vmax=np.amax(rewards), alpha=0.5)
    ## plt.colorbar()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(states[:,0], states[:,1], states[:,2], c=np.squeeze(values),
               marker='o', edgecolors='none', s=35,
               vmin=np.amin(rewards), vmax=np.amax(rewards), alpha=0.5)
    ## plt.colorbar()

    ## if trajs is not None:
    ##     for traj in trajs:
    ##         plt.plot(traj[:,0], traj[:,1], 'r-')
    
    if env is not None:
        ax1.set_xlim( [env.observation_space.low[0], env.observation_space.high[0] ])
        ax1.set_ylim( [env.observation_space.low[1], env.observation_space.high[1] ])
        ax1.set_zlim( [env.observation_space.low[2], env.observation_space.high[2] ])
        ax2.set_xlim( [env.observation_space.low[0], env.observation_space.high[0] ])
        ax2.set_ylim( [env.observation_space.low[1], env.observation_space.high[1] ])
        ax2.set_zlim( [env.observation_space.low[2], env.observation_space.high[2] ])

    if save_plot:
        fig.savefig(filename, format='png')
    else:
        plt.show()


def traj_plot_3d(demo_traj, fitted_traj, xlim=None, ylim=None, zlim=None):
    if type(demo_traj) is list: demo_traj = np.array(demo_traj)
    if len(demo_traj.shape)==2:
        demo_traj = np.expand_dims(demo_traj, axis=0)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #plt.plot(objects[:,0], objects[:,1], 'ko')
    for traj in demo_traj:
        plt.plot(traj[:,0], traj[:,1], traj[:,2], 'r-')

    if fitted_traj is not None:
        if type(fitted_traj) is list: fitted_traj = np.array(fitted_traj)
        if len(fitted_traj.shape)==2:
            fitted_traj = np.expand_dims(fitted_traj, axis=0)
        for traj in fitted_traj:
            plt.plot(traj[:,0], traj[:,1], traj[:,2], 'b-')

    c = []; x_max = []; x_min = []
    c = np.median(traj[:,:3], axis=0)
    x_max = np.amax(traj[:,:3], axis=0)
    x_min = np.amin(traj[:,:3], axis=0)
    r     = max( max(x_max - c), min(c - x_min) )
    
    ## plt.plot(self.start_state[0], self.start_state[1], 'rx')
    ## plt.plot(self.goal_state[0], self.goal_state[1], 'r^')        
    ax.set_xlim( [c[0]-r, c[0]+r] )
    ax.set_ylim( [c[1]-r, c[1]+r] )
    ax.set_zlim( [c[2]-r, c[2]+r] )
    plt.show()
    

##### temp


def traj_objs_plot(traj=None, objects=None, labels=None, states=None, env=None, goal=None, start=None,
                   org_traj=None, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'b']
    markers = ['x', '^']

    if objects is not None:
        objects = np.array(objects)
        if labels is not None:
            l = np.unique(labels)
            new_objs = [[] for _ in range(len(l))]
            for i, label in enumerate(labels):
                new_objs[label].append(objects[i])
            # obects
            for i, obj in enumerate(new_objs):
                obj = np.array(obj)
                ax.scatter(obj[:,0], obj[:,1], obj[:,2] ,c=colors[i], marker=markers[i])
        else:
            ax.scatter(objects[:,0], objects[:,1], objects[:,2] ,c='r', marker=markers[i])
        
    # plot vertices of env
    v = []
    for x in [env.cart_min_pos[0], env.cart_max_pos[0]]:
        for y in [env.cart_min_pos[1], env.cart_max_pos[1]]:
            for z in [env.cart_min_pos[2], env.cart_max_pos[2]]:
                v.append([x,y,z])
    v = np.array(v)
    ax.scatter(v[:,0],v[:,1],v[:,2], c='m', marker='*')

    if goal is not None or env is not None:
        if goal is not None:
            frame = env.get_frame(goal)
        else:
            frame = env.goal_frame
        viz_axis(ax, frame, length=0.2)                   

    if start is not None or env is not None:
        if start is not None:
            frame = env.get_frame(start)
        else:
            frame = env.start_frame
        viz_axis(ax, frame, length=0.2)                   

    
    # traj
    env.verbose=True
    if traj is not None:
        if env is not None:
            
            poses = []
            if start is not None: env.set_start_state(start)
            else: env.set_start_state(traj[0])
            if goal is not None: env.set_goal_state(goal)
            else: env.set_goal_state(traj[-1])
            
            print "------------------------------------------"
            for i, jnt in enumerate(traj):                                
                if env.isValid(jnt) is False:
                    print "{}: invalid joint angle".format(i)
                    print jnt
                frame = env.get_frame(jnt)
                poses.append([frame.p[0], frame.p[1], frame.p[2] ])

                ## print env.get_progress(jnt), frame.M.UnitX()
                if env.get_progress(jnt)>0.2: # or env.get_progress(jnt)<0.1: #0.977:
                    viz_axis(ax, frame, length=0.05)                   

            traj = poses

        if type(traj) is list: traj = np.array(traj)
        ax.plot(traj[:,0],traj[:,1],traj[:,2], 'k-')


    if org_traj is not None and env is not None:
        poses = []
        print "------------------------------------------"
        for i, jnt in enumerate(org_traj):                                
            if env.isValid(jnt) is False:
                print "{}: invalid joint angle".format(i)
            frame = env.get_frame(jnt)
            poses.append([frame.p[0], frame.p[1], frame.p[2] ])
        poses = np.array(poses)
        ax.plot(poses[:,0],poses[:,1],poses[:,2], 'r.')

    if states is not None:
        poses = []
        for i in range(0,len(states),100):
            frame = env.get_frame(states[i])
            poses.append([frame.p[0], frame.p[1], frame.p[2] ])
        poses = np.array(poses)
        ax.scatter(poses[:,0], poses[:,1], poses[:,2], 'ko')

    subgoals = kwargs.get('subgoals', None)
    if subgoals is not None:
        p = np.array(subgoals)
        ax.plot(p[:,0], p[:,1], p[:,2], 'm*', markersize=12)


    ax.set_xlim( [env.cart_min_pos[0], env.cart_max_pos[0]])
    ax.set_ylim( [env.cart_min_pos[1], env.cart_max_pos[1]])
    ax.set_zlim( [env.cart_min_pos[2], env.cart_max_pos[2]] )
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
        
    plt.show()


def viz_axis(ax, frame, length=0.2):
    rx = [ [frame.p[0], frame.p[1], frame.p[2] ] ]
    x = frame.M.UnitX()*length
    rx.append([frame.p[0]+x[0], frame.p[1]+x[1], frame.p[2]+x[2]])
    rx = np.array(rx)
    ax.plot(rx[:,0],rx[:,1],rx[:,2], c='r')
    
    rx = [ [frame.p[0], frame.p[1], frame.p[2] ] ]
    x = frame.M.UnitY()*length
    rx.append([frame.p[0]+x[0], frame.p[1]+x[1], frame.p[2]+x[2]])
    rx = np.array(rx)
    ax.plot(rx[:,0],rx[:,1],rx[:,2], c='g')
    
    rx = [ [frame.p[0], frame.p[1], frame.p[2] ] ]
    x = frame.M.UnitZ()*length
    rx.append([frame.p[0]+x[0], frame.p[1]+x[1], frame.p[2]+x[2]])
    rx = np.array(rx)
    ax.plot(rx[:,0],rx[:,1],rx[:,2], c='b')
    
    
def viz_values_3d(env, states, values, thres=0.5, max_pts=10000):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    states = np.array(states)
    values = np.array(values)

    # start # goal
    ## start = env.get_pos(states[-2])
    ## goal  = env.get_pos(states[-1])
    ## ax.scatter(start[0], start[1], start[2], s=50, c='r', marker='*')
    ## ax.scatter(goal[0], goal[1], goal[2], s=50, c='r', marker='*')

    # sampling
    import random
    ids = random.sample(range(0, len(values)), max_pts)

    states = states[ids]
    values = values[ids]
    ## print np.amax(values), np.amin(values)

    # filter values
    values = (values-np.amin(values))/(np.amax(values)-np.amin(values))
    ids = [i for i, v in enumerate(values) if v > thres]
    print np.shape(ids)
    values = values[ids]
    states = states[ids]
    assert len(values)>0, "zero-size value vector is sampled. lower the threshold!!"
    values = (values-np.amin(values))/(np.amax(values)-np.amin(values))

    poses=[]
    for i, s in enumerate(states):
        poses.append(env.get_pos(s))
    poses = np.array(poses)

    new_values = []
    new_poses  = []
    for i, (p,v) in enumerate(zip(poses, values)):
        if p[2]>-0.05: continue        
        new_values.append(v)
        new_poses.append(p)
    values = np.array(new_values)
    poses  = np.array(new_poses)
    
    scat = ax.scatter(poses[:,0], poses[:,1], poses[:,2], c=values, marker='o')

    ax.set_xlim( [env.cart_min_pos[0], env.cart_max_pos[0]])
    ax.set_ylim( [env.cart_min_pos[1], env.cart_max_pos[1]])
    ax.set_zlim( [env.cart_min_pos[2], env.cart_max_pos[2]] )

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    fig.colorbar(scat)
    plt.show()



def trajs_viz(trajs):

    ## trajs = [traj for i, traj in enumerate(trajs) if (i!=len(trajs)-2) and i!=1 ]
    trajs = np.array(trajs)

    fig = plt.figure(figsize=(12,4))    
    n_dim = len(trajs[0,0])

    import itertools
    colors = itertools.cycle(['r', 'g', 'b', 'm', 'c', 'k', 'y'])
    for i in range(n_dim):
        ax      = fig.add_subplot(n_dim,1,i+1)
        
        for j in range(len(trajs)):
            color = colors.next()
            ax.plot(trajs[j][:,i], color+'-')
    
    plt.show()


def states_viz(env, states, max_pts=1000, traj=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')

    # sampling
    import random
    ids = random.sample(range(0, len(states)), max_pts)
    ids.append(len(states)-2)
    ids.append(len(states)-1)
    states = states[ids]

    poses=[]
    for i, s in enumerate(states):
        poses.append(env.get_pos(s))
    poses = np.array(poses)

    # start # goal
    ax.plot(poses[-2:,0], poses[-2:,1], poses[-2:,2], c='r', marker='*')
    ax.plot(poses[-1:,0], poses[-1:,1], poses[-1:,2], c='b', marker='*')
    ax.scatter(poses[:,0], poses[:,1], poses[:,2], marker='o')

    if traj is not None:
        poses = []
        for i, jnt in enumerate(traj):                                
            frame = env.get_frame(jnt)
            poses.append([frame.p[0], frame.p[1], frame.p[2] ])
        poses = np.array(poses)
        ax.plot(poses[:,0],poses[:,1],poses[:,2], 'r-')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim( [env.cart_min_pos[0], env.cart_max_pos[0]])
    ax.set_ylim( [env.cart_min_pos[1], env.cart_max_pos[1]])
    ax.set_zlim( [env.cart_min_pos[2], env.cart_max_pos[2]] )
   
    plt.show()


def ik_trajs_plot(env, trajs=None, org_trajs=None, subgoals=None):

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if trajs is not None:
        for traj in trajs:
            poses = []
            for i, jnt in enumerate(traj):
                frame = env.get_frame(jnt)
                poses.append([frame.p[0], frame.p[1], frame.p[2] ])
            poses = np.array(poses)
            ax.plot(poses[:,0],poses[:,1],poses[:,2], 'k-')
            
    if org_trajs is not None:
        for traj in org_trajs:
            poses = []
            for i, jnt in enumerate(traj):                                
                frame = env.get_frame(jnt)
                poses.append([frame.p[0], frame.p[1], frame.p[2] ])
            poses = np.array(poses)
            ax.plot(poses[:,0],poses[:,1],poses[:,2], 'r-')
        
    if subgoals is not None:
        p = np.array(subgoals)
        ax.plot(p[:,0], p[:,1], p[:,2], 'm*', markersize=12)
   
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim( [env.cart_min_pos[0], env.cart_max_pos[0]])
    ax.set_ylim( [env.cart_min_pos[1], env.cart_max_pos[1]])
    ax.set_zlim( [env.cart_min_pos[2], env.cart_max_pos[2]] )
   
    plt.show()


def play_traj(env, traj):
    import rospy
    old_state, time = env.reset(), 0
    traj = np.array(traj)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        print np.shape(traj), time
        a = traj[time+1]-traj[time]
        old_state, r, done, _ = env.step(a)
        time += 1
        if time > len(traj)-2 or done:
            env.reset()
            time = 0
            #break
        rate.sleep()


def distPts2Line(A,B,P):
    
    #if type(P) is np.ndarray:
    d = np.linalg.norm(np.cross( P-A , P-B)) / np.linalg.norm(B-A)
    #else:

    return d

def plot_feature_distribution(f_mean_l, f_std_l):

    for i in range(len(f_mean_l)):
        f_mean = f_mean_l[i]
        f_std = f_std_l[i]
        
        # plot over states
        plt.rcParams['image.cmap'] = 'jet'

        ind   = range(len(f_mean))
        width = 0.7

        plt.figure(figsize=(20,8))
        plt.bar(ind, f_mean, width, color='r', yerr=f_std)
        plt.title("The distribution of features [{}]".format(i), fontsize=24)
        plt.xlabel("Indices of features", fontsize=18)
        plt.ylabel("Average feature value", fontsize=18)
        plt.show()
    

## def feature_over_traj(env, trajs, instrs, feature_fn):

##     fig = plt.figure(figsize=(20,8))

##     fss_keys = {}
##     for key in trajs.keys():
##         fss = []
##         for i, (traj, instr) in enumerate(zip(trajs[key], instrs[key])):
##             if i>0: continue

##             env.set_start_state(traj[0])
##             env.set_goal_state(traj[-1])            

##             fs = []
##             for s in traj:
##                 f = feature_fn(env, s, instr[0]).tolist()
##                 fs.append(f)

##             fss.append(fs)
##         print key
##         fss_keys[key] = fss

##     colors = ['r', 'b', 'k', 'm', 'g']
##     for i in range(np.shape(fss_keys[key])[-1]):
##         fig.add_subplot(np.shape(fss_keys[key])[-1],1,i+1)

##         for k, key in enumerate(fss_keys.keys()):            
##             fss = np.array(fss_keys[key])
##             for j in range(len(fss)): 
##                 plt.plot(fss[j,:,i], c=colors[k])

##     plt.show()


def feature_over_traj(env, traj, feature_fn):

    sys.path.insert(0,'..')
    from utils import misc

    if len(traj[0])==6:
        pose_traj = []
        for i in range(len(traj)):
            pose_traj.append( misc.list_rpy2list_quat(traj[i]) )
    else:
        pose_traj = traj

    fig = plt.figure(figsize=(20,8))

    fs = []
    for s in pose_traj:
        f = feature_fn(env, s).tolist()
        fs.append(f)
    fs = np.array(fs)

    colors = ['r', 'b', 'k', 'm', 'g']
    for i in range(len(fs[0])):
        fig.add_subplot(len(fs[0]),1,i+1)
        plt.plot(fs[:,i], c=colors[i])

    plt.show()


def subgoal_marker_pub(waypoints, topic_name, marker_id=0, mesh_resource=None,
                       base_frame_name='base'):
    from visualization_msgs.msg import MarkerArray
    from visualization_msgs.msg import Marker
    import rospy
        
    marker_pub  = rospy.Publisher(topic_name, MarkerArray, queue_size=1)
    rospy.sleep(0.5)
    
    # Set up our waypoint markers
    marker_lifetime     = 0 # 0 is forever
    marker_ns           = 'waypoints'

    if marker_id>=500 and marker_id<600:
        marker_color        = {'r': 0.2, 'g': 0.8, 'b': 1.0, 'a': 1.0}
    elif marker_id>=600:
        marker_color        = {'r': 0., 'g': 1., 'b': 0.0, 'a': 1.0}
    else:
        marker_color        = {'r': 1.0, 'g': 0.2, 'b': 0.2, 'a': 1.0}

    if mesh_resource is not None:
        marker_scale        = 1.0
        marker_color['a']   = 0.8
    else:
        marker_scale        = 0.01

        if marker_id >=1000:
            marker_scale = 0.03
            marker_color = {'r': np.random.uniform(low=0, high=1.),
                            'g': np.random.uniform(low=0, high=1.),
                            'b': np.random.uniform(low=0, high=1.),
                            'a': 1.0}
        
    
    # Define a marker publisher list.
    waypoint_name_list       = list()
    waypoint_name_list       = range(len(waypoints))

    markerArray = MarkerArray()
    markerArray.markers[:] = []
    
    for waypoint in waypoints:
        
        marker                  = Marker()
        marker.ns               = marker_ns
        marker.id               = marker_id
        if mesh_resource is not None:
            marker.type             = Marker.MESH_RESOURCE
            marker.mesh_resource = mesh_resource
        else:
            marker.type             = Marker.SPHERE
        marker.action           = Marker.ADD
        marker.lifetime         = rospy.Duration(marker_lifetime)
        marker.scale.x          = marker_scale
        marker.scale.y          = marker_scale
        marker.scale.z          = marker_scale
        
        marker.color.r          = marker_color['r']
        marker.color.g          = marker_color['g']
        marker.color.b          = marker_color['b']
        marker.color.a          = marker_color['a']
        
        marker.header.frame_id  = base_frame_name
        marker.header.stamp     = rospy.Time.now()
        marker.pose.position.x  = waypoint[0]
        marker.pose.position.y  = waypoint[1]
        marker.pose.position.z  = waypoint[2]
        marker.pose.orientation.x = waypoint[3]
        marker.pose.orientation.y = waypoint[4]
        marker.pose.orientation.z = waypoint[5]
        marker.pose.orientation.w = waypoint[6]
        ## marker.text             = waypoint        
        markerArray.markers.append(marker)
        
        marker_id               = marker_id + 1
    marker_pub.publish(markerArray)



def pose_array_pub(poses, topic_name, base_frame_name='base'):
    import rospy
    from geometry_msgs.msg import PoseArray
        
    pose_pub  = rospy.Publisher(topic_name, PoseArray, queue_size=1)
    rospy.sleep(0.5)

    obj = PoseArray()
    obj.header.frame_id = base_frame_name
    for ps in poses:
        obj.poses.append(ps)
    
    pose_pub.publish(obj)
