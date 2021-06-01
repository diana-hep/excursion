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

N_EVENTS=1000

#setupATLAS
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
asetup 19.2.5.34.2,MCProd,here

#
# Using lepton universality also in the dark sector,
# compute only the ee channel and use a factor of 2 later on to add the muon channel.
#

channel='ee'

#cd $BASE_DIR

tag=$(echo "DMs${MED_TYPE}_${channel}_mR${MASS_ZPRIME}_mDM${MASS_DM}_gQ${G_Q}_gL${G_L}" | sed 's/\./p/g')

jobOptions="MC15.999999.MGPy8EG_N30LO_A14N23LO_${tag}.py"

# If JOs don't exist yet, make them!
if [[ ! -f "${SOURCE_DIR}/share/${jobOptions}" ]]; then
    python ${SOURCE_DIR}/makeJOs_DM.py "${SOURCE_DIR}/share/${jobOptions}"
fi

evnt_dir="${BASE_DIR}/run_${tag}/EVNT"
mkdir -p $evnt_dir
cd ${evnt_dir}

cp "${SOURCE_DIR}/share/MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_ll.py" "${SOURCE_DIR}/share/$jobOptions" "${SOURCE_DIR}/share/MadGraph_param_card_DMsimp_s_spin1.dat" .

#if [[ -f "${outdir}/EVNT.root" ]]; then
#    echo "${outdir}/EVNT.root already exists! Skipping."
#else
Generate_tf.py --ecmEnergy=13000. --maxEvents=${N_EVENTS} --runNumber=999999 --firstEvent=1 --randomSeed=123456 --outputEVNTFile=EVNT.root --jobConfig=${jobOptions}
#fi


cd $BASE_DIR

