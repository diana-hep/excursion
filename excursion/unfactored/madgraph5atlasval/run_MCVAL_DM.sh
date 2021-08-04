batch="true"
checkWidth="false"
overwrite="false"

repo="/home/edreyer/project/edreyer/madgraph5atlasval"

#Set up the validation step
cd "${repo}/source/MCVal"
#source Setup.sh
#make clean
#make


#JO loop
for channel in 'ee'; do # 'mumu'; do

    #for mass in '2p6' '2p7' '2p8' '2p9' '3p0' '3p1' '3p2' '3p3' '3p4' '3p5' '3p6' '3p7' '3p8' '3p9' '4p0' '4p1' '4p2' '4p3' '4p4'; do

    for mass in '0p30' '0p35' '0p40' '0p45' '0p50' '0p55' '0p60' '0p65' '0p70' '0p75' '0p80' '0p85' '0p90' '0p95' '1p00' '1p05' '1p10' '1p15' '1p20' '1p25' '1p30' '1p35' '1p40' '1p45' '1p50' '1p55' '1p60' '1p70' '1p80' '1p90' '2p00' '2p10' '2p20' '2p30' '2p40' '2p50'; do

    #for mass in '1p50' '1p60' '1p70' '1p80' '1p90' '2p00' '2p10' '2p20' '2p30' '2p40' '2p50' '2p60' '2p70' '2p80' '2p90' '3p00' '3p10' '3p20' '3p30' '3p40' '3p50' '3p60' '3p70' '3p80' '3p90' '4p00' '4p10' '4p20' '4p30' '4p40' '4p50' '4p60' '4p70' '4p80' '4p90' '5p00'; do

        massDM='10p0'

        #for massDM in '0p00' '0p05' '0p10' '0p15' '0p20' '0p25' '0p30' '0p35' '0p40' '0p45' '0p50' '0p55' '0p60' '0p65' '0p70' '0p75' '0p80' '0p85' '0p90' '0p95' '1p00' '1p05' '1p10' '1p15' '1p20' '1p25' '1p30' '1p35' '1p40' '1p45' '1p50' '1p55' '1p60'; do

            mediatorType='A'
            #mediatorType='V'

            couplingQuarks='0p1'

            #couplingLeptons='0p1'
            couplingLeptons='0p01'

            for couplingLeptons in '0p001' '0p002' '0p003' '0p004' '0p005' '0p006' '0p007' '0p008' '0p009'; do

                cd ${repo}

		#tag="ee_${mass}_${massDM}_${couplingLeptons}_DM_allLep"
		tag="DMs${mediatorType}_${channel}_mR${mass}_mDM${massDM}_gQ${couplingQuarks}_gL${couplingLeptons}"

		outdir="/home/edreyer/scratch/DM/run_${tag}/MCVAL"
		mkdir -p ${outdir}

		daod="${outdir}/../DAOD/DAOD_TRUTH0.${tag}.root"
		xsec="$(python ${repo}/parseXsec.py  ${outdir}/../EVNT/log.generate)"
		width="$(python ${repo}/parseWidth.py ${outdir}/../EVNT/ParticleData.local.xml)"

		mMed=${mass}
		mMed=${mMed//p/.}
		mMed=`echo "$mMed * 1000" | bc`
		mDM=${massDM}
		mDM=${mDM//p/.}
		mDM=`echo "$mDM * 1000" | bc`
		gQ=${couplingQuarks//p/.}
		gL=${couplingLeptons//p/.}
	   
		if [[ $mediatorType == "V" ]]; then
		    mediatorTypeString="vector"
		elif [[ $mediatorType == "A" ]]; then
		    mediatorTypeString="axial"
		fi

		####################
		#You need to do:
		#    git clone git@github.com:LHC-DMWG/DarkMatterWidthCalculation.git
		# for the following calculation!
		####################

		#echo "python DarkMatterWidthCalculation/wCalc.py --type ${mediatorTypeString} --gQ ${gQ} --gL ${gL} --gDM 1 --mMed ${mMed} --mDM ${mDM}"

		widthAlt="$(python DarkMatterWidthCalculation/wCalc.py --type ${mediatorTypeString} --gQ ${gQ} --gL ${gL} --gDM 1 --mMed ${mMed} --mDM ${mDM})"
		widthAlt=${widthAlt//full mediator width=/}
		widthAlt=${widthAlt//GeV (*)/}

		#full mediator width=7.707e+01 GeV ( 2.96%)

		if [[ ${overwrite} == "true" ]]; then
		    rm -v "${outdir}/events.root"
		fi

		if [[ $checkWidth == "true" ]]; then
		    echo "${mMed} ${mDM} ${width} ${widthAlt} ${gQ} ${gL} 1"
		elif [[ -f "${outdir}/events.root" ]]; then
		    echo "${outdir}/events.root already exists! Skipping."
		elif   [[ $batch == "false" ]]; then
		    ./MC15Validation ${daod} ${mass} ${widthAlt} ${couplingLeptons} ${xsec} ${massDM} ${outdir}

		    #mv "events.root" "${outdir}/events_${tag}.root"
		    echo "${outdir}/events.root"

		elif [[ $batch == "true" ]]; then
		    cmd="sbatch --time=00:10:00 --account=ctb-stelzer --mem-per-cpu=2048M ${repo}/submit_cedar.sh ${repo} MCVAL ${outdir} ${tag} ${mass} ${widthAlt} ${couplingLeptons} ${xsec} ${massDM}"
		    echo ${cmd}
		    #${cmd}

		fi

	done
    done
done

cd ../../
