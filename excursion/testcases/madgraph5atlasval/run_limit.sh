#!/bin/sh

MASS_ZPRIME=$1
WIDTH_PERCENT=$2
OUTPUT_FILE_BASE_NAME=$3
EXP_OR_OBS=$4 # must be "exp" or "obs"
SOURCE_DIR=$5

#cd DileptonReinterpretationProj

echo "================================"
echo "Getting experimental limit      "
echo ""
echo "MASS_ZPRIME is ${MASS_ZPRIME}"
echo "WIDTH_PERCENT is ${WIDTH_PERCENT}"
echo "OUTPUT_FILE_BASE_NAME is ${OUTPUT_FILE_BASE_NAME}"
echo "EXP_OR_OBS is ${EXP_OR_OBS}"
echo "SOURCE_DIR is ${SOURCE_DIR}"
echo ""
echo "================================"

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup "root 6.14.04-x86_64-slc6-gcc62-opt"
asetup AnalysisBase,21.2.51,here

${SOURCE_DIR}/DileptonReinterpretationProj/build/LimitInterpolator -mass ${MASS_ZPRIME} -width ${WIDTH_PERCENT} -channel ll -type ${EXP_OR_OBS} -file ${SOURCE_DIR}/DileptonReinterpretationProj/output/LimitInterpolator_CL95.root -path ${OUTPUT_FILE_BASE_NAME}.root | grep "The ${EXP_OR_OBS} limit" | awk '{print $NF}' | sed 's/fb//' > ${OUTPUT_FILE_BASE_NAME}.txt

#cd ..

