"""

Probablistic Road Map (PRM) Planner

author: Daehyung Park

"""
import copy
import random
import numpy as np
import time

from scipy.stats import norm, beta, skewnorm, truncnorm
import scipy.spatial
import sklearn
from sklearn.neighbors import BallTree, KDTree as kt
import matplotlib.pyplot as plt

import PyKDL
from tqdm import tqdm

from cbn_irl.utils import quaternion as qt
from cbn_irl.utils import misc
import traj_utils as tu
import planning_utils as pu
import dijkstra_planning as dp

show_animation = False
np.random.seed(0)

# pickup
## init_stds = [0.03,0.05,0.05, np.pi/3., np.pi/3., np.pi/3., 0.3] #xyz rx ry rz gripper
MAX_EDGE_LEN = 0.02  # [m] Maximum edge length
## weights = [0.9, 0.1, 0.]

class Node:
    """
    Node class for dijkstra search
    """
    def __init__(self, x, cost, pind):
        self.x = x
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.cost) + "," + str(self.pind)

    
def pose_distance(X, Y, weights=[0.9, 0.01, 0.]):
    """Weighted distance
    weights = [position, orientation, gripper]"""
    if type(X) is not np.ndarray: X = np.array(X)
    if type(Y) is not np.ndarray: Y = np.array(Y)
        
    if len(np.shape(X))==1:
        pos_dist = np.linalg.norm(X[:3]-Y[:3])
        ang_dist = qt.quat_angle(X[3:7], Y[3:7])/np.pi/2.
        grip_dist = abs(X[-1]-Y[-1])
        return pos_dist*weights[0] + ang_dist*weights[1] + grip_dist*weights[2]
    else:
        """X is a 2d array
           Y is a 1d array
        """
        pos_dist = np.linalg.norm(X[:,:3]-Y[:3], axis=-1)
        ang_dist = qt.quat_angle(X[:, 3:7], Y[3:7])/np.pi/2.
        grip_dist = abs(X[:,-1]-Y[-1])
        return pos_dist*weights[0] + ang_dist*weights[1] + grip_dist*weights[2]


## def rpy_pose_distance(X, Y, orient_weight=0.1):
##     """ """
##     new_Y = misc.list_rpy2list_quat(Y)
    
##     if len(np.shape(X))==1:
##         new_X = misc.list_rpy2list_quat(X)
##     else:
##         new_X = []
##         for x in X:
##             new_X.append( misc.list_rpy2list_quat(x) )
    
##     return pose_distance(new_X, new_Y, orient_weight=orient_weight)


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data, leafsize=10):
        self.tree = BallTree(data, leafsize, metric='pyfunc', func=pose_distance)

    def search(self, inp, k=1):
        u"""
        Search NN

        inp: input data, single frame or multi frame

        """
        if type(inp) is list: inp = np.array(inp)

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist
        else:
            dist, index = self.tree.query(inp[np.newaxis,:], k=k)
            return index[0], dist[0]

    def search_in_distance(self, inp, r):
        u"""
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


def get_roadmap(env, knn=5, n_sample=500, traj=None,
                std_traj=None, leafsize=None, viz=False,
                init_stds=[0.03,0.05,0.05, 0.1, 0.1, 0.1, 0.1, 0.2],
                **kwargs):
    u"""Generate roadmap"""

    if traj is None: n_points = 90
    else:            n_points = len(traj)
    if leafsize is None: leafsize=int(knn*1.5)

    if traj is not None:
        env.set_start_state(traj[0])
        env.set_goal_state(traj[-1])

    # Set a straight trajectory or use a demo traj
    if traj is None:
        return NotImplementedError
        ## traj_pts, traj_stds = get_straight_traj(env, init_stds,
        ##                                         n_points=n_points,
        ##                                         default_y_angle=y_angle)
    else:
        traj_pts, traj_stds = get_distribution_on_traj(traj, init_stds=init_stds)

    # Add the traj to samples
    samples = add_traj_points(env, traj_pts)

    # Add uniform random nodes
    #samples = add_random_points(env, n_sample=30, default_y_angle=0, samples=samples, min_dist=min_dist)

    # generate a roadmap
    ## print "generating a roadmap with {} samples".format(len(samples))
    roadmap, skdtree = generate_roadmap(samples, env, MAX_EDGE_LEN, leafsize=leafsize)
    
    connected                 = False
    enable_dikstra            = True
    enable_connectivity_check = True

    if viz:
        from irl_constraints_learning.utils import draw_scene as ds
        sd = ds.SceneDraw('sample_points', frame='/base')
    

    cnt = 0
    while True:
        cnt += 1
        if enable_connectivity_check:
            # check the connectivity and compute groups
            connected, node_groups = check_connectivity(env, roadmap, samples,
                                                        enable_dikstra=enable_dikstra)
            print "Samples {}, groups {}, group sum {}".format(len(samples),
                                                               len(node_groups),
                                                               sum([len(group)
                                                                    for group
                                                                    in node_groups]) )
            if connected: enable_dikstra = False
            ## from IPython import embed; embed(); sys.exit()
        
        if len(node_groups)==1:
            enable_connectivity_check = False
            if len(samples)>n_sample:
                break
            
        # Expand the roadmap using random points
        for i in range(10):
            samples, roadmap = sample_points_rrm(env, traj_pts, traj_stds, samples,
                                                 node_groups, roadmap,
                                                 max_dist=MAX_EDGE_LEN,
                                                 leafsize=leafsize)
        if viz and cnt%5==0:
            sd.pub_points(samples, color=[0,0,1,0.89], num=1000)
            
            
    print "generated a roadmap"
    ## roadmap, skdtree = generate_roadmap(samples, env, MAX_EDGE_LEN, leafsize=leafsize, knn=knn)
    roadmap, skdtree = generate_roadmap_parallel(samples, env, MAX_EDGE_LEN, leafsize, knn)
    ## skdtree = KDTree(samples, leafsize=leafsize)

    return roadmap, np.array(samples).astype(np.float16), skdtree


def get_straight_traj(env, init_stds, n_points=50):
    """Straight line demonstration"""
    start = env.start_state
    goal  = env.goal_state

    dx        = ( goal[:3]-start[:3] ) / float(n_points)
    traj_pts  = [ start     ]
    traj_stds = [ init_stds ]
    dists     = []
    
    for i in range(n_points):

        point = copy.deepcopy(traj_pts[0])
        point[:3] += dx*float(i+1)

        q = qt.slerp(start[3:], goal[3:], float(i+1)/float(n_points))
        point[3] = q[0]
        point[4] = q[1]
        point[5] = q[2]
        point[6] = q[3]

        traj_pts.append( point )
        ## traj_stds.append( list(np.array(init_stds)*(1. + float(n_points-i)/float(n_points))/2.0 ))
        traj_stds.append( list(np.array(init_stds)*np.exp(-1.5*float(i)/float(n_points)) ))

        dists.append(pose_distance(traj_pts[i-1], traj_pts[i]))

    assert np.amax(dists) < MAX_EDGE_LEN, "reference trajectory's resolution {} is too low".format(np.amax(dists))
    return traj_pts, traj_stds


def get_distribution_on_traj(traj, init_stds, coeff=2.):
    """Straight line demonstration """

    n_points = len(traj)

    traj_pts = []
    traj_stds = [] 
    for i in range(n_points):
        traj_pts.append( list(traj[i]) )
        traj_stds.append( list(np.array(init_stds)*max(np.exp(-coeff*float(i)/float(n_points)), 0.04) ))
    return traj_pts, traj_stds


def add_traj_points(env, traj_pts):
    """Add points"""
    samples = []
    for point in traj_pts:
        if env.isValid(point, check_collision=True):
            samples.append(point)    
    return samples


def generate_roadmap(samples, env, max_dist, leafsize=50, knn=None):
    """
    Road map generation

    sample: [m] positions of sampled points (n_samples x n_dim)
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(samples)
    if len(samples)<leafsize: leafsize=len(samples)-1
    skdtree = KDTree(samples, leafsize=leafsize)

    for (i, x) in zip(range(nsample), samples):

        tmp_max_dist=max_dist
        ## dist = env.get_distance(x)
        ## if dist < collision_dist:
        ##     tmp_max_dist=max_dist
        ## else:
        ##     tmp_max_dist=min(dist, max_dist*collision_mult)
        
        try:
            inds, dists = skdtree.search(x, k=leafsize)
        except:
            print "skdtree search failed"
            from IPython import embed; embed(); sys.exit()
            
        edge_id = []
        for ii, (ind, dist) in enumerate(zip(inds, dists)):
            if dist > tmp_max_dist: break # undirected
            if knn is not None and len(edge_id)>=knn: break # directed?
            #nx = samples[ind]
            ## if not is_collision(x, nx, rr, obkdtree, env=env): # or ii<=5:
            edge_id.append(ind)

        # to complement fewer number of edges for vectorized valueiteration
        if knn is not None and len(edge_id) < knn:
            for ii in range(0,len(inds)):
                ## if inds[ii] not in edge_id:
                ##     edge_id.append(inds[ii])
                for ind in edge_id:
                    edge_id.append(ind)
                    if len(edge_id) >= knn: break
                if len(edge_id) >= knn: break

        assert len(edge_id)<=leafsize, "fewer leaves than edges {} (dists={})".format(len(edge_id),
                                                                                      dists[:len(edge_id)] )
        #assert len(edge_id)<=knn, "fewer actions than edges {} (dists={})".format(len(edge_id), dists[:len(edge_id)] )

        road_map.append(edge_id)
        if i%5000 == 0 and i>0: print i
        ## assert len(edge_id)>0, "number of edge of {} node is 0".format(i)

    return road_map, skdtree


def generate_roadmap_parallel(samples, env, max_dist, leafsize, knn):
    """Parallelized roadmap generator """

    n_sample = len(samples)
    leafsize = knn
    if len(samples)<leafsize: leafsize=len(samples)-1

    import sharedmem
    sample_ids = np.arange(n_sample, dtype='i')
    roadmap    = sharedmem.full((n_sample, knn), 0)
    
    # Start multi processing over samples
    with sharedmem.MapReduce() as pool:
        if n_sample % sharedmem.cpu_count() == 0:
            chunksize = n_sample / sharedmem.cpu_count()
        else:
            chunksize = n_sample / sharedmem.cpu_count() + 1
    
        def work(i):
            skdtree        = KDTree(samples, leafsize=leafsize)
            sub_sample_ids = sample_ids[slice (i, i + chunksize)]

            for j, sub_sample_id in enumerate(sub_sample_ids):
                x = samples[sub_sample_id]
                try:
                    inds, dists = skdtree.search(x, k=leafsize)
                except:
                    print "skdtree search failed"
                    sys.exit()

                edge_id = []
                append = edge_id.append
                for ii, (ind, dist) in enumerate(zip(inds, dists)):
                    if dist > max_dist: break # undirected
                    if len(edge_id)>=knn: break # directed?
                    append(ind)

                # to complement fewer number of edges for vectorized valueiteration
                if len(edge_id) < knn:
                    for ii in range(0,len(inds)):
                        #for ind in edge_id:
                        #    edge_id.append(ind)
                        #    if len(edge_id) >= knn: break
                        append(inds[0])
                        if len(edge_id) >= knn: break

                assert len(edge_id)<=leafsize, "fewer leaves than edges {} (dists={})".format(len(edge_id), dists[:len(edge_id)] )

                for k in range(len(edge_id)):
                    roadmap[sub_sample_id][k] = edge_id[k]

        pool.map(work, range(0, n_sample, chunksize))#, reduce=reduce)

    # convert sharedmem array to list
    roadmap = np.array(roadmap).astype(np.uint16) # Unsigned integer (0 to 65535)
    skdtree = None #KDTree(samples, leafsize=leafsize)
    return roadmap.tolist(), skdtree



def add_node(index, group, roadmap):
    """BFS"""
    queue = []
    queue.append(index)
    
    while queue:
        s = queue.pop(0)

        try:
            for i in roadmap[s]:
                if not(i in group):
                    queue.append(i)
                    group.append(i)
        except:
            print i, s, len(roadmap)
    return group
        

def check_connectivity(env, roadmap, states, enable_dikstra=True):
    """Check node connectivity and grouping those"""
    if enable_dikstra:
        traj, _ = dp.dijkstra_planning(env, env.start_state, env.goal_state,
                                       roadmap, states,
                                       distFunc=pose_distance,
                                       verbose=True)
        if traj is not None: connected = True
        else:                connected = False
    else:
        connected = True

    # grouping - TODO: make this faster!!
    node_groups = []
    for i in range(len(states)):
        visited = False
        for group in node_groups:
            if i in group:
                visited = True
                break
        if visited: continue
        node_groups.append([i])
        node_groups[-1] = add_node(i, node_groups[-1], roadmap)
    
    group_sum = sum([len(group) for group in node_groups])
    print "# of groups", [len(group) for group in node_groups]
    print "Sum of groups = {}, States = {}".format( group_sum, len(states) )

    assert group_sum == len(states), "group sum is wrong"
        ## from IPython import embed; embed(); sys.exit()    
    assert group_sum==len(states), "sum {} does not equal to n_samples {}".format(
        group_sum, len(states))
        
    return connected, node_groups




def sample_points_rrm(env, traj_pts, traj_stds, samples, node_groups, roadmap,
                      max_dist=0.005, min_dist=0.001, leafsize=30):

    # random samples
    nearest_samples = []

    if len(node_groups)>1:
        sub_samples = []
        for group in node_groups:
            sub_samples.append( np.array(samples)[group].tolist() )
    else:
        sub_samples = [samples]

    n_points = len(traj_pts)
    p        = np.linspace(1., 0.1, n_points)
    p        /= np.sum(p)
    
    for i, (point, std) in enumerate(zip(traj_pts, traj_stds)):
        # random
        i     = np.random.choice(range(n_points), p=p)
        point = traj_pts[i]
        std   = traj_stds[i]
        ## from IPython import embed; embed(); sys.exit()
        
        rnd = np.array([ truncnorm( (env.observation_space.low[0]-point[0])/std[0],
                                    (env.observation_space.high[0]-point[0])/std[0],
                                    loc=point[0], scale=std[0] ).rvs(1)[0],
                         truncnorm( (env.observation_space.low[1]-point[1])/std[1],
                                    (env.observation_space.high[1]-point[1])/std[1],
                                    loc=point[1], scale=std[1] ).rvs(1)[0],
                         truncnorm( (env.observation_space.low[2]-point[2])/std[2],
                                    (env.observation_space.high[2]-point[2])/std[2],
                                    loc=point[2], scale=std[2] ).rvs(1)[0],
                         0,0,0,1] )
        rnd[3:7] = qt.quat_QuTem(point[3:7], 1, std[3:7])[0]

        for sub_sample in sub_samples:
            # Find nearest node        
            nind = np.argmin(pose_distance(sub_sample, rnd))

            #block
            d_max    = 0.01 # 0.005 
            w_max    = 1.   
            max_dist = MAX_EDGE_LEN
            min_dist = 0.008 #0.009 # 0.017 = 1deg, scale=0.3

            nearest_sample = copy.deepcopy(sub_sample[nind])

            # ang
            org_dist = qt.quat_angle(nearest_sample[3:7], rnd[3:7])/np.pi/2.
            nearest_sample[3:7] = qt.slerp(nearest_sample[3:7], rnd[3:7],
                                           min(w_max/180.*np.pi/org_dist, 1.))
        
            # pos
            unit_vec  = rnd[:3]-nearest_sample[:3]
            unit_vec /= np.linalg.norm(unit_vec)

            d_collision = env.get_distance(nearest_sample)
            if d_collision <= 0: continue
            else:                expandPosDis = d_max
            nearest_sample[:3]+= expandPosDis*unit_vec
            if expandPosDis==0:  continue

            if env.isValid(nearest_sample, check_collision=True) is False:
                continue

            dists = pose_distance( samples+nearest_samples,
                                   np.array( nearest_sample) )
            j = np.argmin( dists )
            pose_dist = dists[j] 
            grip_dist = abs((samples+nearest_samples)[j][7]-nearest_sample[7])
            
            if pose_dist>max_dist or (pose_dist<min_dist and grip_dist<0.5): continue
            nearest_samples.append(nearest_sample)

    if len(nearest_samples)==0: return samples, roadmap

    #expand roadmap
    pre_max_indx = len(samples)-1
    roadmap += [[i+len(samples)] for i in range(len(nearest_samples)) ]
    samples += nearest_samples
    state_inds = np.array(range(len(samples)))

    max_edge = 0
    for s in nearest_samples:
        dists = pose_distance(samples, s)

        try:
            inds  = state_inds[dists<=max_dist]
            dists = dists[inds]

            tmp  = np.argsort(dists)
            inds = np.array(inds)[tmp]
            roadmap[inds[0]] = inds.tolist()            
        except:
            print "Maybe env's observation limit does not match with reference traj"
            from IPython import embed; embed(); sys.exit()
        

        for ii, ind in enumerate(inds):
            if ii==0: continue
            if pre_max_indx<ind: continue
            roadmap[ind].append(inds[0])
            if len(roadmap[ind]) > max_edge:
                max_edge = len(roadmap[ind])

    print len(samples), max_edge, len(nearest_samples)
    return samples, roadmap


## def add_random_points(env, n_sample=500, samples=None, min_dist=0.001):
##     """Sample x,y,z,r,p """    
##     if samples is None: samples = []

##     cnt = 0
##     while cnt<n_sample:# or reached is False:
##         # Random Sampling
##         rnd = np.array([random.uniform(env.observation_space.low[0], env.observation_space.high[0]),
##                         random.uniform(env.observation_space.low[1], env.observation_space.high[1]),
##                         random.uniform(env.observation_space.low[2], env.observation_space.high[2]),
##                         0,0,0,1])
##         M = PyKDL.Rotation.RPY(random.uniform(-np.pi, np.pi),
##                                random.uniform(-np.pi, np.pi),
##                                default_y_angle)            
##         q = M.GetQuaternion()
##         rnd[3] = q[0]
##         rnd[4] = q[1]
##         rnd[5] = q[2]
##         rnd[6] = q[3]

##         if env.isValid(rnd, check_collision=True) is False:
##             continue

##         # Find nearest node        
##         dist = np.amin(pose_distance(samples, rnd))
##         if dist<min_dist: continue

##         samples.append(rnd.tolist())
##         cnt += 1

##     return samples


