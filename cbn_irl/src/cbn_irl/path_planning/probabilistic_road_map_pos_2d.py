"""

Probablistic Road Map (PRM) Planner

author: Atsushi Sakai (@Atsushi_twi)

"""
import copy
import random
import numpy as np
from scipy.stats import norm, beta, skewnorm, truncnorm
import scipy.spatial
import sklearn
from sklearn.neighbors import BallTree, KDTree as kt
import matplotlib.pyplot as plt

import PyKDL
from tqdm import tqdm

import planning_utils as pu
import traj_utils as tu

import dijkstra_planning as dp
import time

# parameter
#N_SAMPLE = 500  # number of sample_points
## N_KNN = 10  # number of edge from one sampled point

show_animation = False
np.random.seed(0)

# reachenv
init_stds = [0.04,0.04] #xy
MAX_EDGE_LEN = 2. # Maximum edge length


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

    
class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data, leafsize=10):
        if type(data) is list: data=np.array(data)
        self.tree = BallTree(data, leafsize)

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
                init_stds = [0.03,0.03], limits=None,
                **kwargs):
    u"""Generate roadmap"""

    if traj is None: n_points = 90
    else:            n_points = len(traj)
    if leafsize is None: leafsize=int(knn*1.5)
        
    # Set a straight trajectory or use a demo traj
    if traj is None:
        samples = add_random_points(env, n_sample=n_sample, min_dist=env.robot_size, limits=limits)
        ## samples = sample_points(start, goal, rr, obstacles,
        ##                         obkdtree, lim, n_sample=n_sample, env=env)
    else:
        env.set_start_state(traj[0])
        env.set_goal_state(traj[-1])
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

    cnt = 0
    while True:
        cnt += 1
        if enable_connectivity_check:
            # check the connectivity and compute groups
            connected, node_groups = check_connectivity(env, roadmap, samples,
                                                        enable_dikstra=enable_dikstra)        
            print "Samples {}, groups {}, group sum {}".format(len(samples), len(node_groups),
                                                               sum([len(group) for group in node_groups]) )
            if connected: enable_dikstra = False
        ## from IPython import embed; embed(); sys.exit()
        
        if len(node_groups)==1:
            enable_connectivity_check = False
            if len(samples)>n_sample:
                break
            
        # Expand the roadmap using random points
        samples = add_random_points(env, n_sample=100, samples=samples, min_dist=env.robot_size, limits=limits)
        roadmap, skdtree = generate_roadmap_parallel(samples, env, MAX_EDGE_LEN, leafsize, knn)
        print cnt, len(samples), len(node_groups)
            
    print "generated a roadmap"
    roadmap, skdtree = generate_roadmap_parallel(samples, env, MAX_EDGE_LEN, leafsize, knn)

    return roadmap, np.array(samples), skdtree


## def get_straight_traj(env, init_stds, n_points=50):
##     """Straight line demonstration"""
##     start = env.start_state
##     goal  = env.goal_state

##     dx        = ( goal-start ) / float(n_points)
##     traj_pts  = [ start     ]
##     traj_stds = [ init_stds ]
##     dists     = []
    
##     for i in range(n_points):

##         point = copy.deepcopy(traj_pts[0])
##         point += dx*float(i+1)

##         traj_pts.append( point )
##         traj_stds.append( list(np.array(init_stds)*np.exp(-1.5*float(i)/float(n_points)) ))

##         dists.append(pose_distance(traj_pts[i-1], traj_pts[i]))

##     assert np.amax(dists) < MAX_EDGE_LEN, "reference trajectory's resolution {} is too low".format(np.amax(dists))
##     return traj_pts, traj_stds


## def get_distribution_on_traj(traj, init_stds, coeff=2.):
##     """Straight line demonstration """

##     n_points = len(traj)

##     traj_pts = []
##     traj_stds = [] 
##     for i in range(n_points):
##         traj_pts.append( list(traj[i]) )
##         traj_stds.append( list(np.array(init_stds)*max(np.exp(-coeff*float(i)/float(n_points)), 0.04) ))
##     return traj_pts, traj_stds


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
        
        try:
            inds, dists = skdtree.search(x, k=leafsize)
        except:
            print "skdtree search failed"
            from IPython import embed; embed(); sys.exit()
            
        edge_id = []
        for ii, (ind, dist) in enumerate(zip(inds, dists)):
            if dist > tmp_max_dist: break # undirected
            if knn is not None and len(edge_id)>=knn: break # directed?
            edge_id.append(ind)

        # to complement fewer number of edges for vectorized valueiteration
        if knn is not None and len(edge_id) < knn:
            for ii in range(0,len(inds)):
                for ind in edge_id:
                    edge_id.append(ind)
                    if len(edge_id) >= knn: break
                if len(edge_id) >= knn: break

        assert len(edge_id)<=leafsize, "fewer leaves than edges {} (dists={})".format(len(edge_id),
                                                                                      dists[:len(edge_id)] )

        road_map.append(edge_id)
        if i%5000 == 0 and i>0: print i

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
    roadmap = np.array(roadmap).astype(int)
    skdtree = None #KDTree(samples, leafsize=leafsize)
    return roadmap.tolist(), skdtree


## def add_node(index, group, roadmap):
##     """DFS"""
##     for i in roadmap[index]:
##         if not(i in group):
##             group.append(i)
##             group = add_node(i, group, roadmap)
##     return group

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
                                       roadmap, states, verbose=True)
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

    assert group_sum == len(states), "group sum is wrong"
        ## from IPython import embed; embed(); sys.exit()

    
    assert group_sum==len(states), "sum {} does not equal to n_samples {}".format(
        group_sum, len(states))
        
    return connected, node_groups



def add_random_points(env, n_sample=500, default_y_angle=0, samples=None, min_dist=0.001, limits=None):
    """Sample x,y,z,r,p """    
    if samples is None: samples = []

    cnt = 0
    while cnt<n_sample:# or reached is False:
        # Random Sampling
        if limits is None:
            rnd = np.array([random.uniform(env.observation_space.low[0], env.observation_space.high[0]),
                            random.uniform(env.observation_space.low[1], env.observation_space.high[1])])
        else:
            rnd = np.array([random.uniform(limits[0][0], limits[0][1]),
                            random.uniform(limits[1][0], limits[1][1])])

        if env.isValid(rnd, check_collision=False) is False:
            continue

        # Find nearest node
        if len(samples) > 1:
            dist = np.amin(np.linalg.norm(samples-rnd, axis=-1))
            if dist<min_dist: continue

        samples.append(rnd.tolist())
        cnt += 1

    return samples




