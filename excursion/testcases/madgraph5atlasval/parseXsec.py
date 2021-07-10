#23:51:32   number      xsec (pb)      xerr (pb)      xmax (pb)
#23:51:32        1     6.1373e-03     8.7401e-06     3.0687e-07

import sys

infile = sys.argv[1]

occurrences = 0
rip = ""

with open(infile) as openfile:
    i = 0
    lines = openfile.readlines()
    for line in lines:
        if ("xsec" in line):
            rip = lines[i+1]
            occurrences = occurrences + 1

        i = i + 1

openfile.close()

if (occurrences == 0):
    print "ERROR: found no occurrence in file!"
if (occurrences > 1): 
    print "ERROR: found more than one occurrence in file!"
else:
    rip = rip.split()[2]
    print rip
