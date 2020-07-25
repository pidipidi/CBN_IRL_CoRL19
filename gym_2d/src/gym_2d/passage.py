import sys
import gym
import gym.spaces
import numpy as np
import copy

def generate_objects(index, env_name, start=None, goal=None, viz=False,
                         return_path=False, path_len=181, **kwargs):
    env = gym.make(env_name)
    env.__init__(start, goal, [], viz=viz)
    xlim = [env.observation_space.low[0], env.observation_space.high[0]]
    ylim = [env.observation_space.low[1], env.observation_space.high[1]]

    d = eval('random{}(xlim, ylim, kwargs.get("w", None), kwargs.get("theta", None), \
    kwargs.get("h", None), return_path, path_len)'.format(index))
        
    if viz:
        objs  = d['objs']
        start = d['start']
        goal  = d['goal']
        path  = d['path']
        
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,8))
        
        if path is not None and len(path)>0:
            path=np.array(path)
            plt.plot(path[:,0], path[:,1], 'r-')
            
        plt.plot(objs[:,0], objs[:,1], 'ko')
        plt.plot(start[0], start[1], 'rx')
        plt.plot(goal[0], goal[1], 'r^')
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ## plt.axis('off')
        ## plt.grid(True)
        plt.show()

        fig.savefig("temp.png", format='png')
        fig.savefig("temp.pdf", format='pdf')


    return d


def random1(xlim, ylim, w=None, theta=None, h=None, return_path=False,
            path_len=181):
    """Sine curve """
    if w is None:     w = np.random.uniform(0.001, 2.)
    if theta is None: theta = np.random.uniform(-np.pi, np.pi)
    if h is None:     h= np.random.uniform(0.05,0.25)

    path_width = 0.25
    dist_from_wall = 0.083

    def get_x(t, a):
        return t + (a*np.cos(t))/np.sqrt(1.+np.cos(t)**2) 
    def get_y(t, a):
        return np.sin(t) - a/np.sqrt(1+np.cos(t)**2) 

    t = np.linspace(0, 2.*np.pi, 360)*w+theta
    tt = np.linspace(0-np.pi/4., 3.*np.pi, 720)*w+theta #obs
    scaler_x = (0.92/max(t))
    scaler_y = (0.92/max(t)) 
    x_offset = path_width/2*np.cos(np.pi/4.)
    y_offset = 0.5+path_width/2*np.cos(np.pi/4.)

    start_x = np.argmin(np.abs(get_x(t, -0.5)*scaler_x+x_offset-0.083))
    end_x   = np.argmin(np.abs(get_x(t, -0.5)*scaler_x+x_offset-0.92))
    
    
    
    left_wall  = np.array([get_x(tt, -1.)*scaler_x+x_offset,
                           get_y(tt, -1.)*scaler_y+y_offset]).T
    path       = np.array([get_x(t, -0.5)[start_x:end_x]*scaler_x+x_offset,
                           get_y(t, -0.5)[start_x:end_x]*scaler_y+y_offset ]).T
    right_wall  = np.array([get_x(tt, 1.)*scaler_x+x_offset,
                            get_y(tt, 1.)*scaler_y+y_offset]).T
    ## np.array([get_x(t, path_width/2)+5,
    ##                        get_y(t, path_width/2)+30.]).T
    path = np.array(path)[ np.linspace(0,len(path)-1,path_len).astype(int) ]
    
    ## x = get_x(t, 0.)
    ## y = get_y(t, 0.)
    ## y_delta = np.cos(t*w+theta)*w*h
    wall_slope = None #np.array([np.ones(len(t))*x[1], y_delta]).T
    ## from IPython import embed; embed(); sys.exit()

    obstacles = left_wall.tolist() + right_wall.tolist()
    labels    = ['left' for _ in range(len(left_wall))] + ['right' for _ in range(len(right_wall))]
    obstacles = np.array(obstacles)

    start = path[0]
    goal  = path[-1]
    ## from IPython import embed; embed(); sys.exit()

    d = {'left_wall_start': left_wall[0], 'left_wall_end': left_wall[-1],
         'right_wall_start': right_wall[0], 'right_wall_end': right_wall[-1],
         'objs': obstacles, 'objs_slope': wall_slope, 'labels': labels,
         'start': start, 'goal': goal,
         'path': path, 'path_width': path_width}

    return d
    



#-------------------------------------------------------------
def make_labelled_trees(objs, labels):
    from cbn_irl.path_planning import probabilistic_road_map as prm

    trees = {}
    objs_in_tree = {}
    for label in np.unique(labels):
        obj = objs[[i for i, l in enumerate(labels) if l==label]]
        trees[label] = prm.KDTree(obj,3) 
        objs_in_tree[label] = obj

    return trees, objs_in_tree
    

def interpolate_path(path, length=50):
    if type(path) is list: path = np.array(path)

    # linear interp
    from scipy.interpolate import interp1d
    t = np.linspace(0, 1, num=len(path), endpoint=True)
    new_t = np.linspace(0, 1, num=length, endpoint=True)

    f = interp1d(t,path[:,0])        
    new_x = f(new_t)
    f = interp1d(t,path[:,1])        
    new_y = f(new_t)
    path = np.array([new_x, new_y]).T

    return path

        
## if __name__ == "__main__":
##     env_name  = 'reaching-v0'
##     _, _, _, _ = generate_objects(21, env_name, return_path=True, viz=True)

    
