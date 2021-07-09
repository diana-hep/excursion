#!/bin/sh

MASS_ZPRIME=$1
MASS_DM=$2
G_Q=$3
G_DM=$4
G_L=$5
MED_TYPE=$6 # expecting "A" for axial vector and "V" for vector mediator Z'

BASE_DIR=${PWD}

#setupATLAS
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup 21.2.65.0,AthDerivation,here

channel='ee'

cd $BASE_DIR

tag=$(echo "DMs${MED_TYPE}_${channel}_mR${MASS_ZPRIME}_mDM${MASS_DM}_gQ${G_Q}_gL${G_L}" | sed 's/\./p/g')
jobOptions="MC15.999999.MGPy8EG_N30LO_A14N23LO_${tag}.py"

daod_dir="${BASE_DIR}/run_${tag}/DAOD"
mkdir -p $daod_dir
cd ${daod_dir}

Reco_tf.py --inputEVNTFile "${daod_dir}/../EVNT/EVNT.root" --outputDAODFile "${tag}.root" --reductionConf TRUTH0



cd $BASE_DIR

