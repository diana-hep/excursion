import sys
import numpy as np

sys.path.append("/ptmp/mpp/prieck/atlasDM/dileptonReso/madgraph5atlasval_complete/madgraph5atlasval/excursion/")

import excursion.utils as utils

_gXMin = 250.0
_gXMax = 6000.0
_gYMin = 0.0
_gYMax = 1600.0

#--------------------------------------------------

def mapFromInterval(x, targetMin, targetMax):
    return targetMin + x * ( targetMax - targetMin )
    
#--------------------------------------------------

def mapXforPlotting(x):
    return mapFromInterval(x, _gXMin, _gXMax)

#--------------------------------------------------

def mapYforPlotting(y):
    return mapFromInterval(y, _gYMin, _gYMax)

#--------------------------------------------------

def invalid_region(x):
    return np.array([False]*len(x))

#--------------------------------------------------

plot_rangedef = np.array([[0.0,1.0,101],[0.0,1.0,101]])
plotG = utils.mgrid(plot_rangedef)
plotX = utils.mesh2points(plotG,plot_rangedef[:,2])

thresholds = [ 0.0 ]

acq_rd = np.array([[0.0,1.0,51],[0.0,1.0,51]])
acqG = utils.mgrid(acq_rd)
acqX = utils.mesh2points(acqG,acq_rd[:,2])

mn_rd = np.array([[0.0,1.0,21],[0.0,1.0,21]])
mnG   = utils.mgrid(mn_rd)
meanX  = utils.mesh2points(mnG,mn_rd[:,2])

# AV benchmark model (mzp, mdm) limits
truthlimits = np.array([
    [ 3578.39, 0 ],
    [ 3575.87, 50 ],
    [ 3578.1, 100 ],
    [ 3581.88, 150 ],
    [ 3578.49, 200 ],
    [ 3587.64, 250 ],
    [ 3591.24, 300 ],
    [ 3606, 350 ],
    [ 3609.92, 400 ],
    [ 3633.75, 450 ],
    [ 3643.4, 500 ],
    [ 3655.71, 550 ],
    [ 3672.71, 600 ],
    [ 3688.69, 650 ],
    [ 3704.96, 700 ],
    [ 3728.79, 750 ],
    [ 3746.94, 800 ],
    [ 3777.25, 850 ],
    [ 3796.19, 900 ],
    [ 3825.06, 950 ],
    [ 3845.72, 1000 ],
    [ 3868.73, 1050 ],
    [ 3894.03, 1100 ],
    [ 3916.35, 1150 ],
    [ 3945.37, 1200 ],
    [ 3977.23, 1250 ],
    [ 4001.03, 1300 ],
    [ 4032.67, 1350 ],
    [ 4064.6, 1400 ],
    [ 4092.29, 1450 ],
    [ 4126.59, 1500 ],
    [ 4159.21, 1550 ],
    [ 4190.13, 1600 ]
    ])

# V benchmark model (mzp, mdm) limits
#scandetails.truthlimits = np.array([
#    [ 682.706, 0 ],
#    [ 682.794, 50 ],
#    [ 684.378, 100 ],
#    [ 689.13, 150 ],
#    [ 704.53, 200 ],
#    [ 735.418, 250 ],
#    [ 781.178, 300 ],
#    [ 838.114, 350 ],
#    [ 904.818, 400 ],
#    [ 984.59, 450 ],
#    [ 1064.14, 500 ],
#    [ 1148.67, 550 ],
#    [ 1247.09, 600 ],
#    [ 1343.94, 650 ],
#    [ 1440.17, 700 ],
#    [ 1535.29, 750 ],
#    [ 1653.17, 800 ],
#    [ 1742.84, 850 ],
#    [ 1830.93, 900 ],
#    [ 1917.74, 950 ],
#    [ 2002.31, 1000 ],
#    [ 2013, 1050 ],
#    [ 2013, 1100 ],
#    [ 2013, 1150 ],
#    [ 2013, 1200 ],
#    [ 2013, 1250 ],
#    [ 2013, 1300 ],
#    [ 2013, 1350 ],
#    [ 2013, 1400 ],
#    [ 2013, 1450 ],
#    [ 2013, 1500 ],
#    [ 2013, 1550 ],
#    [ 2013, 1600 ]
#    ])
#

