#Set up the validation step
cd source/MCVal
setupATLAS
asetup 20.7.3.4,AtlasDerivation,here

#JO loop
for channel in 'mumu' 'ee'; do
    for mass in '500' '1000' '3000'; do

        #Copy output EVNT.root to MCVal area
        cp "../../run_${channel}_${mass}/EVNT.root" "EVNT_${channel}_${mass}.root"

        #Create DAOD_TRUTH0.TRUTH.root file from EVNT.root
        Reco_tf.py --inputEVNTFile "EVNT_${channel}_${mass}.root" --outputDAODFile TRUTH.root --reductionConf TRUTH0

        #Move DAOD_TRUTH0.TRUTH.root to where MCVal wants it
        mv DAOD_TRUTH0.TRUTH.root ../DAOD_TRUTH0_${channel}_${mass}.TRUTH.root

    done
done

#Run validation
source Setup.sh
./Run.sh
