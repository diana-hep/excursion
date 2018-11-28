import numpy
import numpy as np

#the "simulator" -- just a python function for now
threshold = np.log(0.05)
def truth(xv,yv):
    return np.log(analysis(xv,yv))

def xsec(xv,yv):
    return 8.5*np.exp(-0.0025*xv)

def eff(xv,yv):
    return np.tanh(0.005*(xv - (yv + 175)))

def stats(nevents):
    return (1-np.tanh((nevents-1)*5))/2.

def analysis(xv,yv):
    return stats(xsec(xv,yv) * eff(xv,yv))

def invalid_region(x,y):
    return x < y+175

N_ACQGSIZE = 21
N_MEANSIZE = 21
N_PLOTSIZE = 101

acqgrid = numpy.array(np.meshgrid(np.linspace(200,800,N_ACQGSIZE), 
                               np.linspace(0,600,N_ACQGSIZE), sparse=False, indexing='xy'))
mean_grid = numpy.array(np.meshgrid(np.linspace(200,800,N_MEANSIZE), 
                                 np.linspace(0,600,N_MEANSIZE), sparse=False, indexing='xy'))
plot_grid = numpy.array(np.meshgrid(np.linspace(200,800,N_PLOTSIZE), 
                                 np.linspace(0,600,N_PLOTSIZE), sparse=False, indexing='xy'))

#list of points over which to probe possible acquisitions
acqX = acqgrid.transpose().reshape(N_ACQGSIZE*N_ACQGSIZE,2)
acqX = acqX[~invalid_region(acqX[:,0],acqX[:,1])]

#list of points over which to take the mean
meanX = mean_grid.transpose().reshape(N_MEANSIZE*N_MEANSIZE,2)
meanX = meanX[~invalid_region(meanX[:,0],meanX[:,1])]

fixed_init = numpy.array([[300,50],[700,50],[600,200]])
