#!/usr/bin/python

import argparse
import os
import ROOT

arg_parser = argparse.ArgumentParser("")
arg_parser.add_argument("--runDir")
arg_parser.add_argument("--mZPrime")
arg_parser.add_argument("--srcDir")
arg_parser.add_argument("--exp_or_obs", choices = ["exp", "obs"])

args = arg_parser.parse_args()

theoryResults = ROOT.TFile("{}/theoryResults.root".format(args.runDir), "READ")
fidXS_ee      = theoryResults.Get("fidXS").GetVal() * 1e3 # pb -> fb
fidXS_ll      = 2.0 * fidXS_ee
width         = theoryResults.Get("width").GetVal()

mZPrime = float( args.mZPrime ) * 1e3

width_perCent = width / mZPrime * 1e2

experimentalXSLimitOutputBaseName = "{}/experimentalLimit_XS_{}_CLs95".format(os.getcwd(), args.exp_or_obs)

os.system("{}/run_limit.sh {} {} {} {} {}".format(args.srcDir, mZPrime, width_perCent, experimentalXSLimitOutputBaseName, args.exp_or_obs, args.srcDir))

xsLimitFile = open(experimentalXSLimitOutputBaseName + ".txt", "r")
xsLimit     = float( xsLimitFile.readlines()[0] )

mu_limit = xsLimit / fidXS_ll

muLimitFile = open(experimentalXSLimitOutputBaseName.replace("XS", "mu") + ".txt", "w")
muLimitFile.write("{}".format(mu_limit))
muLimitFile.close()





