
MXd =   10000. #DM mass
MY1 =   2000. #mediator mass
gVXd = 0. #vector coupling to DM
gAXd = 1. #axial coupling to DM
gAl11 = 0.02  #axial couplings to electrons
gAl22 = 0.    #axial couplings to muons
gVl11 = 0.    #vector couplings to electrons
gVl22 = 0.    #vector couplings to muons

# include("MC15JobOptions/MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_ll.py")
# locally, while not in release:
import os
include("MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_ll.py")
