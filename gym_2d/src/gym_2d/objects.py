import sys
import gym
import gym.spaces
import numpy as np
import copy

def generate_objects(index, env_name, start=None, goal=None, viz=False,
                     return_path=False,
                     path_len=181):
    env = gym.make(env_name)
    env.__init__(start, goal, [], viz=viz)
    xlim = [env.observation_space.low[0], env.observation_space.high[0]]
    ylim = [env.observation_space.low[1], env.observation_space.high[1]]

    d = eval('obstacles{}(xlim, ylim, start, goal, return_path, path_len)'.format(index))
        
    if viz:
        objs  = d['objs']
        start = d['start']
        goal  = d['goal']
        path  = d['path']
        
        import matplotlib.pyplot as plt        
        if path is not None and len(path)>0:
            path=np.array(path)
            plt.plot(path[:,0], path[:,1], 'r-')
            
        plt.plot(objs[:,0], objs[:,1], 'ko')
        plt.plot(start[0], start[1], 'rx')
        plt.plot(goal[0], goal[1], 'r^')

        plt.text(start[0]-5, start[1]-5, "Start" , fontsize=18)
        plt.text(goal[0]-5, goal[1]-5, "Goal" , fontsize=18 ) 
        
        plt.xlim(xlim)
        plt.ylim(ylim)

        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig('test.pdf', format='pdf')
        fig.savefig('test.png', format='png')
        plt.show()

    return d


def obstacles0(xlim, ylim, start=None, goal=None, return_path=False):

    if start is None: start = [5.,30.]
    if goal is None: goal  = [55.,30.]

    obstacles   = []
    labels = []
    for i in range(xlim[0], xlim[1]):
        obstacles.append([i, ylim[0]+(ylim[1]-ylim[0])*2./3.])
        labels.append('left')
    for i in range(xlim[0], xlim[1]):
        obstacles.append([i, ylim[0]+(ylim[1]-ylim[0])*1./3.])
        labels.append('right')

    obstacles = np.array(obstacles)
            
    return (obstacles, labels), start, goal, None


def obstacles1(xlim, ylim, start=None, goal=None, return_path=False):
    
    if start is None: start = [5.,ylim[1]*5/6]
    if goal is None:  goal  = [55.,ylim[1]/6]

    obstacles   = []
    labels = []
    #0
    for i in range(0, int(xlim[1]*2/3)):
        obstacles.append([i, ylim[1]])
        labels.append('left')
    for i in range(int(xlim[1]*2/3), xlim[1]):
        obstacles.append([i, ylim[1]/3. ])
        labels.append('left')
    for i in range(int(ylim[1]/3), ylim[1]):
        obstacles.append([xlim[1]*2/3, i ])
        labels.append('left')
    #1
    for i in range(0, int(xlim[1]/3)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('right')
    for i in range(int(xlim[1]/3), xlim[1]):
        obstacles.append([i, 0 ])
        labels.append('right')
    for i in range(0, int(ylim[1]*2/3)):
        obstacles.append([xlim[1]/3, i ])
        labels.append('right')        

    obstacles = np.array(obstacles)
            
    return (obstacles, labels), start, goal, None


def obstacles2(xlim, ylim, start=None, goal=None, return_path=False):
    
    if start is None: start = [5.,ylim[1]*5/6]
    if goal is None:  goal  = [50.,ylim[1]/6]

    obstacles   = []
    labels = []
    #0
    for i in range(0, int(xlim[1])):
        obstacles.append([i, ylim[1]])
        labels.append('left')
    for i in range(0, int(ylim[1])):
        obstacles.append([xlim[1]-1, i ])
        labels.append('left')
    #1
    for i in range(0, int(xlim[1]*2/3)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('right')
    for i in range(0, int(ylim[1]*2/3)):
        obstacles.append([xlim[1]*2/3, i ])
        labels.append('right')        

    obstacles = np.array(obstacles)
            
    return (obstacles, labels), start, goal, None


def obstacles3(xlim, ylim, start=None, goal=None, return_path=False):
    
    if start is None: start = [5.,10.]
    if goal is None: goal = [55.,10.]

    obstacles   = []
    labels = []
    #0
    for i in range(int(xlim[1]*1/4), int(xlim[1]*3/4)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('left')

    obstacles = np.array(obstacles)
    
    return (obstacles, labels), start, goal, None

def obstacles4(xlim, ylim, start=None, goal=None, return_path=False):
    
    if start is None: start = [5.,10.]

    obstacles   = []
    labels = []
    #0
    for i in range(int(xlim[1]*1/4), int(xlim[1]*3/4)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('left')

    obstacles = np.array(obstacles)
    
    if goal is None:
        goal  = copy.copy(obstacles[-1])
        goal[1] -= 4
            
    return (obstacles, labels), start, goal, None

# -----------------------------------------------
# Reaching movement -----------------------------
# -----------------------------------------------
def obstacles20(xlim, ylim, start=None, goal=None, return_path=False):
    """ Env. for a touching bar"""

    offset = 4.4
    if start is None: start = [5.,10.]
    if goal is None:  goal  = [int(xlim[1]*3/4),ylim[1]*2/3-offset]

    obstacles   = []
    labels = []
    #0
    for i in range(int(xlim[1]*1/4), int(xlim[1]*3/4)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('left')

    obstacles = np.array(obstacles)

    if return_path:
        path = []
        path.append(start)
        path.append([xlim[1]/4, ylim[1]*2/3.-offset])
        path.append([xlim[1]*3/4., ylim[1]*2/3.-offset])
        path.append(goal)
        print np.shape(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None
    

def obstacles21(xlim, ylim, start=None, goal=None, return_path=False, path_len=181):
    """Environment for a semi-circular path 
    """
    
    if ylim[1] == 1.:
        start = np.array([8./60.,ylim[1]/4])
        goal  = np.array([52./60.,ylim[1]/4])
    else:
        start = np.array([8.,ylim[1]/4])
        goal  = np.array([52.,ylim[1]/4])

    c = (start+goal)/2.
    r = np.linalg.norm(goal-start)/2.*1.2

    obstacles = []
    labels    = []
    #0
    for i in range(0, 181):
        x = c[0] + r*np.cos(np.pi*float(i)/180.0)
        y = c[1] + r*np.sin(np.pi*float(i)/180.0)
        obstacles.append( [x,y] )
        labels.append('left')

    obstacles = np.array(obstacles)

    d = {'wall_center': c, 'wall_radius': r,
         'wall_start': obstacles[0], 'wall_middle': obstacles[len(obstacles)/2], 'wall_end': obstacles[-1],
         'objs': obstacles, 'labels': labels,
         'start': start, 'goal': goal}

    if return_path:
        if ylim[1] == 1.:
            offset = 4.4 / 60.
        else:
            offset = 4.4 
            
        path = []
        c = (start+goal)/2.
        r = np.linalg.norm(goal-start)/2.
        for i in range(0, 181)[::-1]:
            x = c[0] + r*np.cos(np.pi*float(i)/180.0)
            y = c[1] + r*np.sin(np.pi*float(i)/180.0)        
            path.append( [x,y] )

        path = np.array(path)[ np.linspace(0,180,path_len).astype(int) ]
        d['path'] = path     
    return d


def obstacles22(xlim, ylim, start=None, goal=None, return_path=False):
    """Tunnel"""
    
    offset = 4.4
    start = [5.,ylim[1]*5/6]
    goal  = [55.,ylim[1]/6]

    obstacles   = []
    labels = []
    #0
    for i in range(0, int(xlim[1]*2/3)):
        obstacles.append([i, ylim[1]])
        labels.append('left')
    for i in range(int(xlim[1]*2/3), xlim[1]):
        obstacles.append([i, ylim[1]/3. ])
        labels.append('left')
    for i in range(int(ylim[1]/3), ylim[1]):
        obstacles.append([xlim[1]*2/3, i ])
        labels.append('left')
    for i in range(0, int(ylim[1]/3)):
        obstacles.append([xlim[1], i ])
        labels.append('left')
        
    #1
    for i in range(0, int(xlim[1]/3)):
        obstacles.append([i, ylim[1]*2/3])
        labels.append('right')
    for i in range(int(xlim[1]/3), xlim[1]):
        obstacles.append([i, 0 ])
        labels.append('right')
    for i in range(0, int(ylim[1]*2/3)):
        obstacles.append([xlim[1]/3, i ])
        labels.append('right')        

    obstacles = np.array(obstacles)

    if return_path:
        path = []

        path.append(start)
        path.append([start[0], ylim[1]-offset])
        path.append([int(xlim[1]*2/3)-offset, ylim[1]-offset])
        path.append([int(xlim[1]*2/3)-offset, ylim[1]/3-offset])
        path.append([goal[0], ylim[1]/3-offset])
        path.append(goal)
        path = np.array(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None
    


def obstacles23(xlim, ylim, start=None, goal=None, return_path=False):
    """ T path (up)"""
    
    offset = 4.4
    start = [5.,ylim[1]/2 + offset]
    goal  = [55.,ylim[1]/2 + offset]

    obstacles   = []
    labels = []
    #0
    for i in range(0, int(xlim[1]*1/3)):
        obstacles.append([i, ylim[1]/2])
        labels.append('left')
    for i in range(int(ylim[1]/2), int(ylim[1]*3/4)):
        obstacles.append([xlim[1]/3, i])
        labels.append('left')
    for i in range(int(xlim[1]*1/3), int(xlim[1]*2/3)):
        obstacles.append([i, ylim[1]*3/4])
        labels.append('left')
    for i in range(int(ylim[1]/2), int(ylim[1]*3/4)):
        obstacles.append([xlim[1]*2/3, i])
        labels.append('left')
    for i in range(int(xlim[1]*2/3), xlim[1]):
        obstacles.append([i, ylim[1]/2 ])
        labels.append('left')
    obstacles = np.array(obstacles)

    if return_path:
        path = []

        path.append(start)
        path.append([xlim[1]/3-offset, ylim[1]/2+offset])
        path.append([xlim[1]/3-offset, ylim[1]*3/4+offset])
        path.append([xlim[1]*2/3+offset, ylim[1]*3/4+offset])
        path.append([xlim[1]*2/3+offset, ylim[1]/2+offset])
        path.append(goal)
        path = np.array(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def obstacles24(xlim, ylim, start=None, goal=None, return_path=False):
    """ T path"""
    
    offset = 4.4
    start = [5.,ylim[1]/2 - offset]
    goal  = [55.,ylim[1]/2 - offset]

    obstacles   = []
    labels = []
    #0
    for i in range(0, int(xlim[1]*1/3)):
        obstacles.append([i, ylim[1]/2])
        labels.append('left')
    for i in range(int(ylim[1]/2), int(ylim[1]*3/4)):
        obstacles.append([xlim[1]/3, i])
        labels.append('left')
    for i in range(int(xlim[1]*1/3), int(xlim[1]*2/3)):
        obstacles.append([i, ylim[1]*3/4])
        labels.append('left')
    for i in range(int(ylim[1]/2), int(ylim[1]*3/4)):
        obstacles.append([xlim[1]*2/3, i])
        labels.append('left')
    for i in range(int(xlim[1]*2/3), xlim[1]):
        obstacles.append([i, ylim[1]/2 ])
        labels.append('left')
    obstacles = np.array(obstacles)

    if return_path:
        path = []

        path.append(start)
        path.append([xlim[1]/3+offset, ylim[1]/2-offset])
        path.append([xlim[1]/3+offset, ylim[1]*3/4-offset])
        path.append([xlim[1]*2/3-offset, ylim[1]*3/4-offset])
        path.append([xlim[1]*2/3-offset, ylim[1]/2-offset])
        path.append(goal)
        path = np.array(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None
    

def obstacles25(xlim, ylim, start=None, goal=None, return_path=False):
    """ ] path"""

    offset = 4.4
    start = [5.,ylim[1]*4/5.]
    goal  = [xlim[1]*3/4.-offset, ylim[1]/4 + offset]

    obstacles   = []
    labels = []
    #0
    for i in range(int(xlim[1]/2), int(xlim[1]*3/4.)):
        obstacles.append([i, ylim[1]*3/4.])
        labels.append('left')
    for i in range(int(ylim[1]*3/4.), int(ylim[1]/4.), -1):
        obstacles.append([xlim[1]*3/4., i])
        labels.append('left')
    for i in range(int(xlim[1]/2), int(xlim[1]*3/4)):
        obstacles.append([i, ylim[1]/4])
        labels.append('left')
    obstacles = np.array(obstacles)

    if return_path:
        path = []
        path.append(start)
        path.append([xlim[1]/2, ylim[1]*3/4-offset])
        path.append([xlim[1]*3/4-offset, ylim[1]*3/4-offset])
        path.append(goal)
        path = np.array(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def obstacles26(xlim, ylim, start=None, goal=None, return_path=False):
    """ Env. for narrow ] path"""

    offset = 4.4
    start = [5.,ylim[1]*4/5.]
    goal  = [xlim[1]*4/5.-offset,ylim[1]/2.]

    obstacles   = []
    labels = []
    #0
    for i in range(int(xlim[1]/4), int(xlim[1]*4/5.)):
        obstacles.append([i, ylim[1]/2.+offset])
        labels.append('left')
    for i in range(int(ylim[1]/2.+offset), int(ylim[1]/2.-offset), -1):
        obstacles.append([xlim[1]*4/5., i])
        labels.append('left')
    for i in range(int(xlim[1]/4), int(xlim[1]*4/5)):
        obstacles.append([i, ylim[1]/2.-offset])
        labels.append('left')
    obstacles = np.array(obstacles)

    if return_path:
        path = []
        path.append(start)
        path.append([xlim[1]/4-offset, ylim[1]/2])
        path.append(goal)
        path = np.array(path)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def obstacles27(xlim, ylim, start=None, goal=None, return_path=False):
    """ Env. for triangle (UP)
    """
    
    offset = 4.4
    if start is None: start = [5.-offset*15./29.155,  20.+offset*25./29.155 ]
    if goal is None: goal = [55.+offset*15./29.155,  20.+offset*25./29.155 ]

    obstacles = []
    labels    = []
    
    #0
    for i in range(0,5):
        obstacles.append([i,20])
        labels.append('left')        
    for x, y in zip( np.linspace(5., 30, 30, endpoint=True), np.linspace(20, 35, 30, endpoint=True) ):
        obstacles.append([x,y])
        labels.append('left')
    for x, y in zip( np.linspace(30., 55, 30, endpoint=True), np.linspace(35., 20., 30, endpoint=True) ):
        obstacles.append([x,y])
        labels.append('left')
    for i in range(55,60):
        obstacles.append([i,20])
        labels.append('left')        

    obstacles = np.array(obstacles)

    if return_path:
        path = []
        path.append(start)
        path.append([30., 35.+offset])
        path.append(goal)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def obstacles28(xlim, ylim, start=None, goal=None, return_path=False):
    """ Env. for triangle (down)
    """
    
    offset = 4.4
    if start is None: start = [5.+offset*15./29.155,  20.-offset*25./29.155 ]
    if goal is None: goal = [55.-offset*15./29.155,  20.-offset*25./29.155 ]

    obstacles = []
    labels    = []
    
    #0
    for i in range(0,5):
        obstacles.append([i,20])
        labels.append('left')        
    for x, y in zip( np.linspace(5., 30, 30, endpoint=True), np.linspace(20, 35, 30, endpoint=True) ):
        obstacles.append([x,y])
        labels.append('left')
    for x, y in zip( np.linspace(30., 55, 30, endpoint=True), np.linspace(35., 20., 30, endpoint=True) ):
        obstacles.append([x,y])
        labels.append('left')
    for i in range(55,60):
        obstacles.append([i,20])
        labels.append('left')        

    obstacles = np.array(obstacles)

    if return_path:
        path = []
        path.append(start)
        path.append([30., 35.-offset])
        path.append(goal)
        path = interpolate_path(path, length=180)
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


# -----------------------------------------------
# Circulate movement -----------------------------
# -----------------------------------------------
    
def obstacles8(xlim, ylim, start=None, goal=None, return_path=False):
    """Environment for a circular path with exterior demo path
    """
    offset = 4.4
    c = np.array([sum(xlim), sum(ylim)])/2.
    r = 20.

    obstacles   = []
    labels = []
    #0
    for i in range(0, 361):
        x = c[0] + r*np.cos(np.pi*float(i)/180.0)
        y = c[1] + r*np.sin(np.pi*float(i)/180.0)
        obstacles.append( [x,y] )
        labels.append('left')
    obstacles = np.array(obstacles)


    r = r+offset
    x = c[0] + r*np.cos(np.pi*float(5)/180.0)
    y = c[1] + r*np.sin(np.pi*float(5)/180.0)        
    start = np.array([x, y])
    x = c[0] + r*np.cos(np.pi*float(355)/180.0)
    y = c[1] + r*np.sin(np.pi*float(355)/180.0)        
    goal = np.array([x, y])


    if return_path:
        path = []
        for i in range(5, 355)[::-1]:
            x = c[0] + r*np.cos(np.pi*float(i)/180.0)
            y = c[1] + r*np.sin(np.pi*float(i)/180.0)        
            path.append( [x,y] )
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def obstacles9(xlim, ylim, start=None, goal=None, return_path=False):
    """Environment for a circular path with interior demo path
    """
    offset = 4.4
    c = np.array([sum(xlim), sum(ylim)])/2.
    r = 25.

    obstacles   = []
    labels = []
    #0
    for i in range(0, 361):
        x = c[0] + r*np.cos(np.pi*float(i)/180.0)
        y = c[1] + r*np.sin(np.pi*float(i)/180.0)
        obstacles.append( [x,y] )
        labels.append('left')
    obstacles = np.array(obstacles)



    r = r-offset
    x = c[0] + r*np.cos(np.pi*float(5)/180.0)
    y = c[1] + r*np.sin(np.pi*float(5)/180.0)        
    start = np.array([x, y])
    x = c[0] + r*np.cos(np.pi*float(355)/180.0)
    y = c[1] + r*np.sin(np.pi*float(355)/180.0)        
    goal = np.array([x, y])


    if return_path:
        path = []
        for i in range(5, 355)[::-1]:
            x = c[0] + r*np.cos(np.pi*float(i)/180.0)
            y = c[1] + r*np.sin(np.pi*float(i)/180.0)        
            path.append( [x,y] )
            
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


#------------------------------------------------------------------------------------
def generate_rnd_objects(index, env_name, viz=False, return_path=False, **kwargs):
    env = gym.make(env_name)
    env.__init__(None, None, [])
    xlim = [env.observation_space.low[0], env.observation_space.high[0]]
    ylim = [env.observation_space.low[1], env.observation_space.high[1]]

    d = eval('random{}(xlim, ylim, kwargs.get("w", None), kwargs.get("theta", None), \
    kwargs.get("h", None), return_path)'.format(index))
        
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
        plt.axis('off')
        ## plt.grid(True)
        plt.show()

        fig.savefig("temp.png", format='png')
        fig.savefig("temp.pdf", format='pdf')


    return d


def random1(xlim, ylim, w=None, theta=None, h=None, return_path=False):
    """Sine curve """
    if w is None:     w = np.random.uniform(0.001, 2.)
    if theta is None: theta = np.random.uniform(-np.pi, np.pi)
    if h is None:     h= np.random.uniform(3,15)

    path_width = 15.
    dist_from_wall = 5.

    t = np.linspace(0, 2.*np.pi, 360)
    x = t*(60./(2.*np.pi))
    y = np.sin(t*w+theta)*h + 30.+path_width/2.
    y_delta = np.cos(t*w+theta)*w*h
    
    left_wall  = np.array([x, y]).T
    right_wall = np.array([x, y-path_width]).T
    path       = np.array([x[8:-8], y[8:-8]-dist_from_wall]).T
    wall_slope = np.array([np.ones(len(t))*x[1], y_delta]).T

    obstacles = left_wall.tolist() + right_wall.tolist()
    labels    = ['left' for _ in range(len(x))] + ['right' for _ in range(len(x))]
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
    


def random2(xlim, ylim, w=None, theta=None, h=None, return_path=False):
    """Spiral """
    if w is None:     w = np.random.uniform(0.001, 3.)
    if theta is None: theta = np.random.uniform(-np.pi, np.pi)
    if h is None:     h= np.random.uniform(3,15)

    path_width = 15.
    dist_from_wall = 4.4


    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r

    t = np.linspace(0, 2.*np.pi, 360)
    x = t*(60./(2.*np.pi))
    y = np.sin(t*w+theta)*h + 30.
    
    left_wall  = np.array([x, y]).T
    right_wall = np.array([x, y-path_width]).T
    path       = np.array([x[8:-8], y[8:-8]-dist_from_wall]).T

    obstacles = left_wall.tolist() + right_wall.tolist()
    labels    = ['left' for _ in range(len(x))] + ['right' for _ in range(len(x))]
    obstacles = np.array(obstacles)

    start = path[0]
    goal  = path[-1]
    print np.shape(obstacles), np.shape(path), np.shape(left_wall)
    ## from IPython import embed; embed(); sys.exit()

    if return_path:
        return (obstacles, labels), start, goal, path
    else:
        return (obstacles, labels), start, goal, None


def random3(xlim, ylim, w=None, theta=None, h=None, return_path=False):
    """Sine curve """
    if w is None:     w = np.random.uniform(0.001, 2.)
    if theta is None: theta = np.random.uniform(-np.pi, np.pi)
    if h is None:     h= np.random.uniform(3,15)

    path_width = 15.
    dist_from_wall = 5.

    def get_x(t, a, h=1.):
        return t + (a*np.cos(t))/np.sqrt(1.+np.cos(t)**2) 
    def get_y(t, a):
        return np.sin(t) - a/np.sqrt(1+np.cos(t)**2) 

    t = np.linspace(0, 2.*np.pi, 360)*w+theta
    tt = np.linspace(0-np.pi/4., 3.*np.pi, 720)*w+theta #obs
    scaler_x = (55./max(t))
    scaler_y = (55./max(t)) #(60./(2.*np.pi))
    #scaler = (60./(2.*np.pi))
    ## width  = 0.5*scaler
    ## scaler /= (path_width/width)
    x_offset = path_width/2*np.cos(np.pi/4.)
    y_offset = 30+path_width/2*np.cos(np.pi/4.)

    start_x = np.argmin(np.abs(get_x(t, -0.5)*scaler_x+x_offset-5.))
    end_x   = np.argmin(np.abs(get_x(t, -0.5)*scaler_x+x_offset-55.))
    
    
    
    left_wall  = np.array([get_x(tt, -1.)*scaler_x+x_offset,
                           get_y(tt, -1.)*scaler_y+y_offset]).T
    path       = np.array([get_x(t, -0.5)[start_x:end_x]*scaler_x+x_offset,
                           get_y(t, -0.5)[start_x:end_x]*scaler_y+y_offset ]).T
    right_wall  = np.array([get_x(tt, 1.)*scaler_x+x_offset,
                            get_y(tt, 1.)*scaler_y+y_offset]).T
    ## np.array([get_x(t, path_width/2)+5,
    ##                        get_y(t, path_width/2)+30.]).T

    
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

    
