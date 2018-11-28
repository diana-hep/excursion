from skimage import measure
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .. import utils

def contour_3d(v, rangedef, level, alpha = None, facecolors = None, edgecolors = None):
    verts, faces, normals, values = measure.marching_cubes(v, level=level, step_size=1)
    true = rangedef[:,0]+(rangedef[:,1]-rangedef[:,0])*np.divide(1.,rangedef[:,2]-1)*verts
    mesh = Poly3DCollection(true[faces])

    if alpha: mesh.set_alpha(alpha)
    if facecolors: mesh.set_facecolors(facecolors)
    if edgecolors: mesh.set_edgecolors(edgecolors)
    return mesh


def plot_current_estimate(ax, gp, X, y, scandetails, funcindex, view_init  = (70,-45)):
    denseGrid  = utils.mgrid(scandetails.plot_rangedef)
    denseX = utils.mesh2points(denseGrid,scandetails.plot_rangedef[:,2])

    prediction, prediction_std = gp.predict(denseX, return_std=True)
    ax.scatter(denseX[:,0],denseX[:,1],denseX[:,2], c = prediction, alpha = 0.05)

    for val,c in zip(scandetails.thresholds,['r','g','y']):
        vals  = prediction.reshape(*map(int,scandetails.plot_rangedef[:,2]))
        mesh = contour_3d(vals,scandetails.plot_rangedef,val,alpha=0.1, facecolors=c, edgecolors=c)
        ax.add_collection3d(mesh)

    truthy = scandetails.truth(denseX)
    for val,c in zip(scandetails.thresholds,['k','grey','blue']):
        vals  = truthy.reshape(*map(int,scandetails.plot_rangedef[:,2]))
        mesh = contour_3d(vals,scandetails.plot_rangedef,val,alpha=0.1, facecolors=c, edgecolors=c)
        ax.add_collection3d(mesh)

    scatplot = ax.scatter(X[:,0],X[:,1],X[:,2], c = 'r', s = 100, alpha = 0.2)

    # scatplot = ax.scatter(X[:,0],X[:,1],X[:,2], c = Y, alpha = 0.05, s = 200)
    ax.set_xlim(scandetails.plot_rangedef[0][0],scandetails.plot_rangedef[0][1])
    ax.set_ylim(scandetails.plot_rangedef[1][0],scandetails.plot_rangedef[1][1])
    ax.set_zlim(scandetails.plot_rangedef[2][0],scandetails.plot_rangedef[2][1])
    ax.view_init(*view_init)
