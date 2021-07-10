#! /bin/bash -l
#
#  This is a template for a batch job which you can use to submit 
#
# See also this link on how to submit to batch queues
# https://twiki.atlas-canada.ca/bin/view/AtlasCanada/ATLASLocalRootBase2#Batch_Jobs
#
#

# site configurations for batch $ALRB_localConfigDir/siteConfigBatch.txt
#----------------------------------------------------------
# Compute Canada site setup for ATLAS
#----------------------------------------------------------

# for change in SLURM that broke --export=NONE
unset TMPDIR
if [ -z $HOME ]; then
   export HOME=`\echo ~`
fi

module load singularity
source /project/atlas/Tier3/AtlasUserSiteSetup.sh

# Reference: https://docs.computecanada.ca/wiki/Running_jobs#Accounts_and_projects

# If you have _more_ than one allocation, configure your ~/.bashrc on the
#  login machine to have these lines (and login again)
#      export SLURM_ACCOUNT=<def-someuser>
#      export SBATCH_ACCOUNT=$SLURM_ACCOUNT
#      export SALLOC_ACCOUNT=$SLURM_ACCOUNT

# Instead of putting the time limit in your batch file,
#  submit with sbatch --time=<timelimit>
# see sbatch --help for other nice things you can add
# and also read the reference link above.


#----------------------------------------------------------


# Environment variables to pass on
export ALRB_testPath=",,,,"
export ALRB_CONT_SWTYPE="singularity"
export ALRB_lcgenvVersion="HEAD"
export ALRB_CONT_RUNPAYLOAD="source ${1}/run_${2}_batch.sh ${3} ${4} ${5} ${6} ${7} ${8} ${9}"
export ALRB_CONT_CMDOPTS=" -B /project:/project -B /scratch:/scratch"
export ALRB_CONT_PRESETUP="hostname -f; date; id -a"
export ALRB_localConfigDir="/project/atlas/Tier3/cedar"
export RUCIO_ACCOUNT="edreyer"
export FRONTIER_SERVER="(serverurl=http://frontier-atlas.lcg.triumf.ca:3128/ATLAS_frontier)(serverurl=http://frontier-atlas1.lcg.triumf.ca:3128/ATLAS_frontier)(serverurl=http://frontier-atlas2.lcg.triumf.ca:3128/ATLAS_frontier)(serverurl=http://frontier-atlas3.lcg.triumf.ca:3128/ATLAS_frontier)(serverurl=http://lcgft-atlas.gridpp.rl.ac.uk:3128/frontierATLAS)(serverurl=http://lcgvo-frontier03.gridpp.rl.ac.uk:3128/frontierATLAS)(serverurl=http://lcgvo-frontier02.gridpp.rl.ac.uk:3128/frontierATLAS)(serverurl=http://lcgvo-frontier01.gridpp.rl.ac.uk:3128/frontierATLAS)(proxyurl=http://lcg-adm1.sfu.computecanada.ca:3128)(proxyurl=http://lcg-adm2.sfu.computecanada.ca:3128)(proxyurl=http://kraken01.westgrid.ca:3128)(proxyurl=http://atlasbpfrontier.fnal.gov:3127)(proxyurl=http://atlasbpfrontier.cern.ch:3127)"
export SITE_NAME="CA-SFU-T2"

# ideally setupATLAS is defined by the site admins.  Just in case ....
alias | \grep -e "setupATLAS" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    if [ ! `typeset  -f setupATLAS > /dev/null` ]; then
	function setupATLAS
	{
            if [ -d  /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase ]; then
		export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
		source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh $@
		return $?
            else
		\echo "Error: cvmfs atlas repo is unavailable"
		return 64
            fi
	}
    fi
fi

# setupATLAS -c <container> which will run and also return the exit code
#  (setupATLAS is source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh)
setupATLAS -c slc6
exit $?
