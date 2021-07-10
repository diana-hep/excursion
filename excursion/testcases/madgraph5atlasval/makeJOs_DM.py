import sys

### parse the options from name of JO file:
#MC15.999999.MGPy8EG_N30LO_A14N23LO_DMs${mediatorType}_${channel}_mR${mass}_mDM${massDM}_gQ${couplingQuarks}_gL${couplingLeptons}.py

name = sys.argv[1]

parsedMedMass = ""
parsedLepCoupling = ""

for bit in name.split("_"):
    bit = bit.replace(".py","")
    if   "mR"  in bit: parsedMedMass     = bit.replace("mR","").replace("p",".")
    elif "mDM" in bit: parsedDMMass      = bit.replace("mDM","").replace("p",".")
    elif "gQ"  in bit: parsedQrkCoupling = bit.replace("gQ","").replace("p",".")
    elif "gL"  in bit: parsedLepCoupling = bit.replace("gL","").replace("p",".")

if(parsedMedMass == ""):
    print "ERROR: cannot determine mediator mass!"
    quit()
else:
    parsedMedMass = str(float(parsedMedMass)*1000)

if(parsedDMMass == ""):
    print "ERROR: cannot determine DM mass!"
    quit()
else:
    parsedDMMass = str(float(parsedDMMass)*1000)

if(parsedLepCoupling == ""):
    print "ERROR: cannot determine lepton coupling!"
    quit()

isElectron = ("ee" in name)
if( (not isElectron) and ("mumu" not in name) ):
    print "ERROR: cannot determine whether it is electron or muon channel!"
    quit()

isVector   = ("DMsV" in name)
if( (not isVector) and ("DMsA" not in name) ):
    print "ERROR: cannot determine whether it is axial or vector!"
    quit()

massDM       = parsedDMMass
massMediator = parsedMedMass


gVector_DM   = "1" if isVector else "0"
gAxial_DM    = "0" if isVector else "1"


gVector_q   = parsedQrkCoupling if isVector else "0"

gVector_el  = parsedLepCoupling if (isVector) else "0"
gVector_mu  = parsedLepCoupling if (isVector) else "0"
gVector_tau = parsedLepCoupling if (isVector) else "0"


gAxial_q    = parsedQrkCoupling if (not isVector) else "0"

gAxial_el   = parsedLepCoupling if (not isVector) else "0"
gAxial_mu   = parsedLepCoupling if (not isVector) else "0"
gAxial_tau  = parsedLepCoupling if (not isVector) else "0"


### write the file:
f=open(name,"w+")
f.write("MXd   = %s  #DM mass\n"                       % massDM)
f.write("MY1   = %s  #mediator mass\n"                 % massMediator)
f.write("gVXd  = %s  #vector coupling to DM\n"         % gVector_DM)
f.write("gAXd  = %s  #axial coupling to DM\n"          % gAxial_DM)

f.write("gVd11 = %s  #vector couplings to quarks:\n"   % gVector_q)
f.write("gVu11 = %s\n"                                 % gVector_q)
f.write("gVd22 = %s\n"                                 % gVector_q)
f.write("gVu22 = %s\n"                                 % gVector_q)
f.write("gVd33 = %s\n"                                 % gVector_q)
f.write("gVu33 = %s\n"                                 % gVector_q)

f.write("gVl11 = %s  #vector coupling to electrons\n"  % gVector_el)
f.write("gVl22 = %s  #vector coupling to muons\n"      % gVector_mu)
f.write("gVl33 = %s  #vector coupling to tauons\n"     % gVector_tau)

f.write("gAd11 = %s  #axial couplings to quarks:\n"    % gAxial_q)
f.write("gAu11 = %s\n"                                 % gAxial_q)
f.write("gAd22 = %s\n"                                 % gAxial_q)
f.write("gAu22 = %s\n"                                 % gAxial_q)
f.write("gAd33 = %s\n"                                 % gAxial_q)
f.write("gAu33 = %s\n"                                 % gAxial_q)

f.write("gAl11 = %s  #axial coupling to electrons\n"   % gAxial_el)
f.write("gAl22 = %s  #axial coupling to muons\n"       % gAxial_mu)
f.write("gAl33 = %s  #axial coupling to tauons\n"      % gAxial_tau)

f.write("\n")

f.write("# include(\"MC15JobOptions/MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_ll.py\")\n")
f.write("# locally, while not in release:\n")
f.write("import os\n")
f.write("include(\"MadGraphControl_MGPy8EG_N30LO_A14N23LO_dmA_ll.py\")\n")
f.close()
