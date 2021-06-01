
MXd =   10000. #DM mass
MY1 =   200. #mediator mass
gVXd = 0. #vector coupling to DM
gAXd = 1. #axial coupling to DM
gAd11 = 0.02 #axial couplings to quarks
gAu11 = 0.02
gAd22 = 0.02
gAu22 = 0.02
gAd33 = 0.02
gAu33 = 0.02
gVd11 = 0. #vector couplings to quarks
gVu11 = 0.
gVd22 = 0.
gVu22 = 0.
gVd33 = 0.
gVu33 = 0.

xptj = 10.

# include("MC15JobOptions/MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_jj_varptcut.py")
# locally, while not in release:
import os
include(os.environ['DIJETEVGENPARENTDIR']+"/MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_jj_varptcut.py")
