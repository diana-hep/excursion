control="MadGraph_control_SingleVectorLQ_U1_DrellYan.py"
param='MadGraph_param_card_SingleVectorLQ_U1_DrellYan.py'

#Setup for EVNT generation
setupATLAS
asetup 19.2.5.34.2,MCProd,here

#JO loop
for channel in 'ee'; do
    for mass in '500' '1000' '3000'; do

        jobOptions="MC15.999999.MadGraphPythia8EvtGen_A14NNPDF23LO_tchan_U1_sbLQ${channel}_M${mass}.py"

        #Initialize work area
        mkdir "run_${channel}_${mass}"
        cd "run_${channel}_${mass}"
        cp "../share/$control" "../share/$jobOptions" "../share/$param" .

        #Create EVNT file
        Generate_tf.py --ecmEnergy=13000. --maxEvents=10000 --runNumber=999999 --firstEvent=1 --randomSeed=123456 --outputEVNTFile=EVNT.root --jobConfig=${jobOptions}

        cd ..
    done
done
