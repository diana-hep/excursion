#!/bin/bash
#. /etc/bashrc

outdir=${1}
jobOptions=${2}

cd ${outdir}
pwd
setupATLAS
asetup 19.2.5.34.2,MCProd,here

EXE="Generate_tf.py --ecmEnergy=13000. --maxEvents=10000 --runNumber=999999 --firstEvent=1 --randomSeed=123456 --outputEVNTFile=EVNT.root --jobConfig=${jobOptions}"
echo "++ Launch "${EXE}
${EXE}
