#!/bin/bash

#
#SBATCH --output=./slurmRun_%j.log
#SBATCH --error=./slurmRun_%j.log
#
#SBATCH -J dileptonLimitCalc
#
#SBATCH --mail-type=NONE
#

echo "==================================================================="
echo "Running limit calculation for dilepton Dark Matter reinterpretation"
echo "==================================================================="

echo ""
echo "mZPrime is ${mZPrime}"
echo "mDM is ${mDM}"
echo "g_q is ${g_q}"
echo "g_DM is ${g_DM}"
echo "g_l is ${g_l}"
echo "medType is ${medType}"
echo "jobId is ${jobId}"
echo "exp_or_obs is ${exp_or_obs}"
echo ""

alias python='/usr/bin/python3'

SourceDir=${SLURM_SUBMIT_DIR}/..
WorkDir=run_${jobId}

mkdir ${WorkDir}
cd ${WorkDir}

${SourceDir}/run_EVNT_DM_singlePoint.sh $mZPrime $mDM $g_q $g_DM $g_l $medType $SourceDir
${SourceDir}/run_DAOD_DM_singlePoint.sh $mZPrime $mDM $g_q $g_DM $g_l $medType
${SourceDir}/run_MCVAL_DM_singlePoint.sh $mZPrime $mDM $g_q $g_DM $g_l $medType $SourceDir

runDir=$(ls | grep ^run_)

echo "Now get XS and width."

root -l -q -b "${SourceDir}/GetTheoryXSecAndWidth.C(\"${runDir}/MCVAL/events.root\", \"${runDir}/theoryResults.root\")"

#
# Note that below the EXPECTED limit is chosen
#

echo "Now finalise limit calc."

python ${SourceDir}/finaliseLimitCalc.py --srcDir ${SourceDir} --runDir ${runDir} --mZPrime ${mZPrime} --exp_or_obs $exp_or_obs



