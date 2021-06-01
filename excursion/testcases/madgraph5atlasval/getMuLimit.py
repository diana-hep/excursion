#!/usr/bin/python

#
# Determine signal strength upper limit for Dark Matter Z' mediator simplified model
# based on the ATLAS Run 2 dilepton resonance search
#

import sys, argparse, os, ROOT
import numpy as np

def getMuLimit(x = np.empty(0), mediator_type = "V"):
    
    argParser = argparse.ArgumentParser( description = "DM Simp dilepton resonance interpretation" )
    argParser.add_argument("--mZPrime", help = "Z\' mediator mass [GeV]", type = float)
    argParser.add_argument("--mDM", help = "Dark Matter mass [GeV]", type = float)
    argParser.add_argument("--g_q", help = "Z\' - quark coupling strength", type = float)
    argParser.add_argument("--g_DM", help = "Z\' - dark matter coupling strength", type = float)
    argParser.add_argument("--g_l", help = "Z\' - charged lepton coupling strength", type = float)
    argParser.add_argument("--mediator_type", help = "Mediator Type - vector (V) or axial vector (A) ", type = str, choices = ["V", "A"] )
    
    if len(sys.argv) > 1:
        args          = argParser.parse_args()
        mZPrime_all   = np.array([args.mZPrime])
        mDM_all       = np.array([args.mDM])
        g_q_all       = np.array([args.g_q])
        g_DM_all      = np.array([args.g_DM])
        g_l_all       = np.array([args.g_l])
        mediator_type = args.mediator_type
        
    elif x.size > 0:
        if mediator_type != "V" and mediator_type != "A":
            print("Error. Mediator type is {} but must be either 'A' (axial vector) or 'V' (vector).".format(mediator_type))
            return -1.0
        mZPrime_all = x[:,0]
        mDM_all     = x[:,1]
        # tmp hack
        g_q_all     = [] # x[:,2]
        g_DM_all    = [] # x[:,3]
        g_l_all     = [] # x[:,4]
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
    #     * Maybe include the GetTheoryXSecAndWidth.C and run_obsLimit.sh steps in the batch jobs as well
    #   * have another loop checking all N seconds whether all jobs are done,
    #     then collect the limits and return them
    #
    #
    
    mu_limit_all = np.array([])

    for iParameterPoint in range(len(mZPrime_all)):

        mZPrime = mZPrime_all[iParameterPoint]
        mDM     = mDM_all[iParameterPoint]
        # tmp hack
        g_q     = 0.1  # g_q_all[iParameterPoint]
        g_DM    = 1.0  # g_DM_all[iParameterPoint]
        g_l     = 0.01  # g_l_all[iParameterPoint]
        # tmp hack end
        
        
        
        mZPrime_rounded = round(mZPrime * 1e-3, 2) * 1e3
        mDM_rounded     = round(mDM * 1e-3, 2) * 1e3

        options = "{} {} {} {} {} {}".format(mZPrime_rounded * 1e-3, mDM_rounded * 1e-3, g_q, g_DM, g_l, mediator_type)

             
        os.system("./run_EVNT_DM_singlePoint.sh {}".format(options))
        os.system("./run_DAOD_DM_singlePoint.sh {}".format(options))
        os.system("./run_MCVAL_DM_singlePoint.sh {}".format(options))
        
        mZp__dir  = mZPrime_rounded * 1e-3
        mDM__dir  = mDM_rounded * 1e-3
        g_q__dir  = g_q
        g_l__dir  = g_l
        directory = "run_DMs{}_ee_mR{}_mDM{}_gQ{}_gL{}".format(mediator_type, mZp__dir, mDM__dir, g_q__dir, g_l__dir)
        directory = directory.replace(".", "p")
        
        os.system("root -l -q -b \"GetTheoryXSecAndWidth.C(\\\"{}/MCVAL/events.root\\\", \\\"{}/theoryResults.root\\\")\"".format(directory, directory))
        
        theoryResults = ROOT.TFile("{}/theoryResults.root".format(directory), "READ")
        fidXS_ee      = theoryResults.Get("fidXS").GetVal() * 1e3 # pb -> fb
        fidXS_ll      = 2.0 * fidXS_ee
        width         = theoryResults.Get("width").GetVal()
        
        width_perCent = width / mZPrime * 1e2
        
        obsLimitOutputFileBaseName="output/obsLimit_{}_CL95".format(directory.replace("run_", "").replace("_ee", ""))
        os.system("./run_obsLimit.sh {} {} {}".format(mZPrime, width_perCent, obsLimitOutputFileBaseName))
        obsLimitFile = open("DileptonReinterpretationProj/{}.txt".format(obsLimitOutputFileBaseName), "r")
        limit_obs_str = obsLimitFile.readline()
        limit_obs     = float("".join(a for a in limit_obs_str if a.isalnum() or a == "."))
        
        mu_limit = limit_obs / fidXS_ll

        mu_limit_all = np.append( mu_limit_all, np.array([mu_limit]) )

    print(mu_limit_all)
    
    return mu_limit_all


#--------------------------------------------------

if __name__ == "__main__":
    getMuLimit()

