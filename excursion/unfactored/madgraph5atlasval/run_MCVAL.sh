#Set up the validation step
cd source/MCVal
setupATLAS
asetup 20.7.3.4,AtlasDerivation,here
source Setup.sh
make clean
make

#JO loop
for channel in 'mumu' 'ee'; do
    for mass in '500' '1000' '3000'; do

        #Run validation
        ./MC15Validation DAOD_TRUTH0_${channel}_${mass}.TRUTH.root
        mv events.root events_${channel}_${mass}.root

    done
done
