import numpy as np

def cube_points(c,wid):
    """ Creates a list of points for plotting a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid]) 
    p.append([c[0]+wid,c[1]-wid,c[2]-wid]) 
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) 
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid]) 
    p.append([c[0]+wid,c[1]-wid,c[2]+wid]) 
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    #vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) 
    p.append([c[0]-wid,c[1]+wid,c[2]+wid]) 
    p.append([c[0]-wid,c[1]+wid,c[2]-wid]) 
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid]) 
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T
