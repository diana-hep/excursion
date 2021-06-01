#!/bin/bash
#. /etc/bashrc

outdir=${1}
tag=${2}

cd ${outdir}
pwd
setupATLAS
asetup 20.7.3.4,AtlasDerivation,here

EXE="Reco_tf.py --inputEVNTFile ${outdir}/../EVNT/EVNT.root --outputDAODFile "${tag}.root" --reductionConf TRUTH0"
echo "++ Launch "${EXE}
${EXE}
