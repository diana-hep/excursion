#!/usr/bin/python

#
# Determine signal strength upper limit for Dark Matter Z' mediator simplified model
# based on the ATLAS Run 2 dilepton resonance search
#

import sys, argparse, os, time  #,pyroot #ROOT
import numpy as np

_gXMin = 250.0
_gXMax = 6000.0
_gYMin = 0.0
_gYMax = 1600.0

_gMZPrimeMin = _gXMin
_gMZPrimeMax = _gXMax
_gMDMMin     = _gYMin
_gMDMMax     = _gYMax



def mapFromUnitInterval(xx, xMin, xMax):
    return np.array( [ xMin + ( xMax - xMin ) * x for x in xx ] )

#--------------------------------------------------

def shapeMuLimit(muLimit):
    muLimit = np.log( muLimit )
    return muLimit

#--------------------------------------------------

def getLatestJobId(username):
    tmpSqueueFileName = "tmpSqueueFile.txt"
    os.system("squeue | grep " + username + " | head -n 1 | awk '{print $1}' > " + tmpSqueueFileName)
    tmpSqueueFile = open(tmpSqueueFileName, "r")
    return int( tmpSqueueFile.readlines()[0] )
    

#--------------------------------------------------

def getMuLimitBatch(x = np.empty(0), mediator_type = "A", exp_or_obs = "exp", submitDirTag = "_run2402_av_2RBFwhite", doGlScan = False):
    print('hello!')
    #argParser = argparse.ArgumentParser( description = "DM Simp dilepton resonance interpretation" )
    #argParser.add_argument("--mZPrime", help = "Z\' mediator mass [GeV]", type = float)
    #argParser.add_argument("--mDM", help = "Dark Matter mass [GeV]", type = float)
    #argParser.add_argument("--g_q", help = "Z\' - quark coupling strength", type = float)
    #argParser.add_argument("--g_DM", help = "Z\' - dark matter coupling strength", type = float)
    #argParser.add_argument("--g_l", help = "Z\' - charged lepton coupling strength", type = float)
    #argParser.add_argument("--mediator_type", help = "Mediator Type - vector (V) or axial vector (A) ", type = str, choices = ["V", "A"] )
    #argParser.add_argument("--slurmUser", help = "Slurm user name", default = "prieck")
    #print(argParser)
    #args = argParser.parse_args()

    
    # if len(sys.argv) > 1:
    #    mZPrime_all   = np.array([args.mZPrime])
    #    mDM_all       = np.array([args.mDM])
    #    g_q_all       = np.array([args.g_q])
    #    g_DM_all      = np.array([args.g_DM])
    #    g_l_all       = np.array([args.g_l])
    #    mediator_type = args.mediator_type
        
    if x.size > 0:
        if mediator_type != "V" and mediator_type != "A":
            print("Error. Mediator type is {} but must be either 'A' (axial vector) or 'V' (vector).".format(mediator_type))
            return -1.0
        # tmp hack
        mZPrime_all = mapFromUnitInterval( x[:,0], _gMZPrimeMin, _gMZPrimeMax )
        g_q_all     = [] # x[:,2]
        g_DM_all    = [] # x[:,3]
        if not doGlScan:
            mDM_all     = mapFromUnitInterval( x[:,1], _gMDMMin, _gMDMMax )
            g_l_all     = [] # x[:,4]
        else:
            mDM_all = []
            g_l_all = np.array( [ 10**( 4.0 * ( xg - 1.0 ) ) for xg in x[:,1] ] )
            
        # tmp hack end
        
        
    else:
        print("Error. Input parameters missing.")
        return -1.0
        
    #
    # ToDo:
    #
    #   * Allow to vary g_DM
    #
    #   * Make sure precision is sufficient for all parameters
    #


    #
    #
    #
    # Turn the following part into a parallel work done on the batch system.
    #
    #   * create a working directory where all results go,
    #     adjust scripts used below accordingly
    #   * loop over all rows of x
    #   * keep track of the created directories
    #   * submit the jobs (each job running the 3 shell scripts)
    #     * Also include the GetTheoryXSecAndWidth.C and run_obsLimit.sh steps in the batch jobs as well,
    #       using a dedicated python script
    #   * have another loop checking all N seconds whether all jobs are done,
    #     then collect the limits and return them
    #
    #
    
    mu_limit_all = np.array([])

    baseDir = os.getcwd()
    submitDir = baseDir + "/submitDir"
    if submitDirTag != "":
        submitDir += "_" + submitDirTag
    if not os.path.exists(submitDir):
        os.mkdir(submitDir)
    os.chdir(submitDir)

    nLimits = len(mZPrime_all)

    tmpFileName = "nRunDirs.txt"
    os.system("ls . | grep -c run_ > {}".format(tmpFileName))
    tmpFile = open(tmpFileName, "r")
    nRunDirs = int(tmpFile.readlines()[0])
    tmpFile.close()
    os.system("rm {}".format(tmpFileName))
    
    #
    # Submit limit calculations as batch jobs
    #
    submitStrings = []
    submitIds     = []
    tmpSqueueFileName = "tmpSqueueFile.txt"
    slurmUser = "iem244"
    for iParameterPoint in range( nLimits ):

        g_q     = 0.1  # g_q_all[iParameterPoint]
        g_DM    = 1.0  # g_DM_all[iParameterPoint]
        mZPrime = mZPrime_all[iParameterPoint]
        #
        # AV benchmark model
        #
        if not doGlScan:
            g_l = 0.1  # g_l_all[iParameterPoint]
            mDM = mDM_all[iParameterPoint]
        else:
            mDM = 10e3
            g_l = round(g_l_all[iParameterPoint], 4)

        mZPrime_rounded = round(mZPrime * 1e-3, 2) * 1e3
        mDM_rounded     = round(mDM * 1e-3, 2) * 1e3
        
        submitStrings.append("sbatch -p standard --export=mZPrime={},mDM={},g_q={},g_DM={},g_l={},medType='{},jobId={},exp_or_obs={}' ../runLimitCalcOnBatch.sh".format(round(mZPrime_rounded * 1e-3, 3), round(mDM_rounded * 1e-3, 3), g_q, g_DM, g_l, mediator_type, iParameterPoint + nRunDirs, exp_or_obs))
        os.system( submitStrings[-1] )
        time.sleep(0.1)
        submitIds.append( getLatestJobId(slurmUser) )
        
    
    print("Waiting for batch jobs to finish...")
    
    #
    # Wait for batch jobs to finish
    #
    while True:
        #
        # check for broken jobs and resubmit if need be
        #
        for iRun in range( nLimits ):
            logFile = "slurmRun_{}.log".format(submitIds[iRun])
            if not os.path.isfile(logFile):
                continue
            os.system("tail -n 1 {} > {}".format(logFile, tmpSqueueFileName))
            tmpSqueueFile = open(tmpSqueueFileName, "r")
            if "AttributeError: 'TObject' object has no attribute 'GetVal'" in tmpSqueueFile.readlines()[-1]:
                os.system("rm {}".format(logFile))
                os.system("rm -rf run_{}".format(iRun + nRunDirs))
                os.system( submitStrings[iRun] )
                submitIds[iRun] = getLatestJobId(args.slurmUser)
        #
        # check if all jobs are finished
        #
        os.system("squeue | grep dilep | grep {} > {}".format(args.slurmUser, tmpSqueueFileName))
        tmpSqueueFile = open(tmpSqueueFileName, "r")
        if len( tmpSqueueFile.readlines() ) == 0:
            break
        time.sleep(1)
    
    time.sleep(3)
    #
    # Collect limits
    #
    print("cwd is {}".format(os.getcwd())) # tmp check
    limits = []
    for iParameterPoint in range( nLimits ):
        limitFileName = "run_{}/experimentalLimit_mu_{}_CLs95.txt".format(iParameterPoint + nRunDirs, exp_or_obs)
        print("Collecting limit {} from {}".format(iParameterPoint, limitFileName))
        limitFile = open(limitFileName, "r")
        limit = float( limitFile.readlines()[0] )
        limit = shapeMuLimit( limit )
        limits.append( limit )
    
    os.chdir(baseDir)
        
    return limits


#--------------------------------------------------

if __name__ == "__main__":
    getMuLimitBatch()

