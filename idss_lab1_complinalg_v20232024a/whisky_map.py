import numpy as np
import matplotlib.pyplot as plt

# NB: YOU DO NOT NEED TO UNDERSTAND THIS CODE

# plot the distillery map
def map_coords(os):
    # OS grip to lat, lon (approx linear transform)
    transform = np.array([[ 1.65384733e-05, -8.44279903e-07, -7.91490157e+00],
                        [ 2.27706834e-07,  9.08229683e-06,  4.97404249e+01],
                        [ 0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
    # convert locations to homogenous form [(x,y,1), (x,y,1), ...]
    hlocations = np.concatenate([os, np.ones((len(os), 1))], axis=1)

    # apply the transformation
    transformed = np.dot(hlocations, transform.T)
    return transformed[:,:-1]



from matplotlib.patches import Polygon



# load the coastline data
import pickle
with open("data/gb_coastline_high.dat", "rb") as f:
    coastline = pickle.load(f)

def map_box(ax, x1,y1,x2,y2):
    tl = map_coords(np.array([[x1, y1]]))[0]
    br = map_coords(np.array([[x2, y2]]))[0]
    
    ax.plot([tl[0],br[0],br[0], tl[0], tl[0]], [tl[1], tl[1], br[1], br[1], tl[1]], c='r', ls=':')
    ax.add_patch(Polygon(
        np.array([[tl[0],br[0],br[0], tl[0], tl[0]], [tl[1], tl[1], br[1], br[1], tl[1]]]).T,
        facecolor='r', alpha=0.2))
    

def draw_map(locations, distilleries, map_attribute=None):
    # new figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # draw the islands
    for islands in coastline:
        ax.add_patch(Polygon(np.array(islands), alpha=0.5))

    transformed = map_coords(locations)
    if map_attribute is not None:
        # draw the distilleries, interpolating
        # the variable onto a grid
        import scipy.interpolate
        mx, my = np.mgrid[-7.8:-0.5:200j, 54:61.0:200j]
        gridded = scipy.interpolate.griddata(transformed[:,:2], 
                                            map_attribute,
                                            (mx,my), method='nearest')
                                            
        img = ax.imshow(gridded.T,extent=(-7.8, -0.5,  54.0, 61.0), origin='top')
        
    ax.scatter(transformed[:, 0], transformed[:, 1], c='C1', zorder=1000, s=20)


    # Add a few labels (too many and they would all overlap)
    for i in range(7, len(distilleries), 10):
        ax.text(transformed[i, 0], transformed[i, 1], distilleries[i], fontsize=8)

    # add key landmarks
    ax.text(-4.292649, 55.873571, "BOYD ORR", fontsize=10, color="white")
    ax.scatter(-4.292649, 55.873571, s=40, c='C3')

    # crop the map to the relevant area
    
    ax.set_frame_on(False)
    ax.set_xlim(-7.8, -0.5)
    ax.set_xlabel("Longitude (deg)")
    ax.set_xlabel("Latitude (deg)")
    ax.set_xlim(-7.8, -0.5)
    
    ax.set_aspect(1.0)
    ax.set_ylim(54.5, 61)
    return ax