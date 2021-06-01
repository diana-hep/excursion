#!/bin/sh

MASS_ZPRIME=$1
MASS_DM=$2
G_Q=$3
G_DM=$4
G_L=$5
MED_TYPE=$6 # expecting "A" for axial vector and "V" for vector mediator Z'
if [ -z "$7" ]; then
    SOURCE_DIR=$PWD
else
    SOURCE_DIR=$7
fi

BASE_DIR=${PWD}

#cd "${BASE_DIR}/source/MCVal"


#setupATLAS
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
rcSetup Base,2.0.21

channel='ee'

tag=$(echo "DMs${MED_TYPE}_${channel}_mR${MASS_ZPRIME}_mDM${MASS_DM}_gQ${G_Q}_gL${G_L}" | sed 's/\./p/g')

mcVal_dir="${BASE_DIR}/run_${tag}/MCVAL"
mkdir -p ${mcVal_dir}

daod="${mcVal_dir}/../DAOD/DAOD_TRUTH0.${tag}.root"
xsec="$(python ${SOURCE_DIR}/parseXsec.py  ${mcVal_dir}/../EVNT/log.generate)"
width="$(python ${SOURCE_DIR}/parseWidth.py ${mcVal_dir}/../EVNT/ParticleData.local.xml)"

${SOURCE_DIR}/source/MCVal/MC15Validation ${daod} ${MASS_ZPRIME} ${width} ${G_L} ${xsec} ${MASS_DM} ${mcVal_dir}

cd $BASE_DIR

