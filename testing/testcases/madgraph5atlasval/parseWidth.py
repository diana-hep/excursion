#<particle id="5000001" name=" y1" spinType="3" chargeType="0" colType="0"
#          m0="1.000e+03" mWidth="2.982e+01" mMin="5.971e+01" mMax="0.000e+00" tau0="6.60521e-15">

import sys

infile = sys.argv[1]

occurrences = 0
rip = ""

with open(infile) as openfile:
    i = 0
    lines = openfile.readlines()
    for line in lines:
        if ("y1" in line):
            rip = lines[i+1]
            occurrences = occurrences + 1

        i = i + 1

openfile.close()

if (occurrences == 0):
    print "ERROR: found no occurrence in file!"
if (occurrences > 1): 
    print "ERROR: found more than one occurrence in file!"
else:
    rip = rip.split()[1]
    rip = rip.replace("mWidth=","")
    rip = rip.replace("\"","")
    print rip
