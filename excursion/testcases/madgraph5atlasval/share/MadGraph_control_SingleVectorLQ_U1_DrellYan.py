import re
import os
import math
import subprocess

from MadGraphControl.MadGraphUtils import *

nevents = 10000
mode = 0
mass=0.500000e+03
channel="mumu"
gsmuL=-1
gseL=-1
gbmuL=-1
gbeL=-1

JOname = runArgs.jobConfig[0]

matches = re.search("M([0-9]+).*\.py", JOname)
if matches is None:
    raise RuntimeError("Cannot find mass string.")
else:
    mass = float(matches.group(1))

if "sbLQmumu" in JOname:
    channel="mumu"
    gsmuL=1.0
    gseL=0.0
    gbmuL=1.0
    gbeL=0.0

elif "sbLQee" in JOname:
    channel="ee"
    gsmuL=0.0
    gseL=1.0
    gbmuL=0.0
    gbeL=1.0

else:
    raise RuntimeError("Cannot find coupling string.")

test=[999999]

fcard = open('proc_card_mg5.dat','w')
if runArgs.runNumber in test and channel=="mumu":
    fcard.write("""
    import model VectorLQ_U1_UFO\n
    define p = p b b~
    define j = j b b~
    generate p p > mu+ mu- NP==2
    output VectorLQSingleProduction""")
    fcard.close()

elif runArgs.runNumber in test and channel=="ee":
    fcard.write("""
    import model VectorLQ_U1_UFO\n
    define p = p b b~
    define j = j b b~
    generate p p > e+ e- NP==2
    output VectorLQSingleProduction""")
    fcard.close()

else:
    raise RuntimeError("runNumber %i not recognised in these jobOptions."%runArgs.runNumber)

beamEnergy = -999
if hasattr(runArgs, 'ecmEnergy'):
    beamEnergy = runArgs.ecmEnergy / 2.
else:
    raise RuntimeError("No center of mass energy found.")

process_dir = new_process()
extras = {'pdlabel': "'lhapdf'",
          'ktdurham': '1.0'}

try:
    os.remove('run_card.dat')
except OSError:
    pass

build_run_card(run_card_old=get_default_runcard(proc_dir=process_dir), run_card_new='run_card.dat',
               nevts=nevents, rand_seed=runArgs.randomSeed, beamEnergy=beamEnergy, extras=extras)

if os.path.exists("param_card.dat"):
    os.remove("param_card.dat")


param_card_name = 'MadGraph_param_card_SingleVectorLQ_U1_DrellYan.py'
param_card = subprocess.Popen(['get_files', '-data', param_card_name])
param_card.wait()
if not os.access(param_card_name, os.R_OK):
    print 'ERROR: Could not get param card'
elif os.access('param_card.dat',os.R_OK):
    print 'ERROR: Old param card in the current directory.  Dont want to clobber it.  Please move it first.'
else:
    oldcard = open(param_card_name, 'r')

    newcard = open('param_card.dat', 'w')

    for line in oldcard:
        if 'mLQ' in line:
            newcard.write('  9000002 %e # mLQ\n' % (mass))
        elif 'gbmuL' in line:
            newcard.write('    2 %e # gbmuL\n' % (gbmuL))
        elif 'gbeL' in line:
            newcard.write('    3 %e # gbeL\n' % (gbeL))
        elif 'gsmuL' in line:
            newcard.write('    4 %e # gsmuL\n' % (gsmuL))
        elif 'gseL' in line:
            newcard.write('    5 %e # gseL\n' % (gseL))
        else:
            newcard.write(line)
    oldcard.close()
    newcard.close()

print_cards()

runName = 'run_01'
process_dir = new_process()
generate(run_card_loc='run_card.dat',
         param_card_loc='param_card.dat',
         mode=mode,
         proc_dir=process_dir,
         run_name=runName)

arrange_output(run_name=runName, proc_dir=process_dir, outputDS=runName + '._00001.events.tar.gz', lhe_version=3,
               saveProcDir=True)

include("MC15JobOptions/Pythia8_A14_NNPDF23LO_EvtGen_Common.py")
include("MC15JobOptions/Pythia8_MadGraph.py")

#### Shower
evgenConfig.description = 'Single-production vector LQ to DrellYan'
evgenConfig.keywords+=['BSM', 'exotic', 'leptoquark', 'scalar']
evgenConfig.generators+=["MadGraph", "Pythia8", "EvtGen"]
evgenConfig.process = 'pp -> ll'
evgenConfig.contact = ["Etienne Dreyer <etienne.dreyer@cern.ch>"]
evgenConfig.inputfilecheck = runName
runArgs.inputGeneratorFile=runName+'._00001.events.tar.gz'
