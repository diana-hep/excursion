import os

indir="/afs/cern.ch/work/e/edreyer/public/madgraph5atlasval/source/MCVal/events_DM/"
outdir="/afs/cern.ch/work/e/edreyer/public/madgraph5atlasval/source/MCVal/events_DM/"
channels=["mumu","ee"]
masses=["500","1000","2000"]
couplings=["0p02"]
#variables=["n_l1","pdgid_l1","e_l1","px_l1","py_l1","pz_l1","pt_l1","eta_l1","phi_l1","n_l2","pdgid_l2","e_l2","px_l2","py_l2","pz_l2","pt_l2","eta_l2","phi_l2","m_ll"]
variables=["pt_l1","pt_l2","eta_l1","eta_l2","phi_l1","phi_l2","m_ll"]
massString={
            "500": "0p5",
            "1000": "1p0",
            "2000": "2p0",
           }
titlex={
        "m_ll":  "m_{ll} [GeV]",
        "pt_l1": "p_{T} (l1) [GeV]",
        "pt_l2": "p_{T} (l2) [GeV]",
        "phi_l1": "#phi (l1)",
        "phi_l2": "#phi (l2)",
        "eta_l1": "#eta (l1)",
        "eta_l2": "#eta (l2)",
       }
minx={
     "eta_l1": "-4",
     "eta_l2": "-4",
     "phi_l1": "-3.5",
     "phi_l2": "-3.5",
    }
maxx={
     "eta_l1": "4",
     "eta_l2": "4",
     "phi_l1": "3.5",
     "phi_l2": "3.5",
    }
channellabel={
              "mumu": "#mu#mu",
              "ee": "ee",
             }
linecolor={
           "500":  8,
           "1000": 9,
           "3000": 2,
          }

for channel in channels:
    for mass in masses:
        for coupling in couplings:
            for var in variables:
                card="card_%s_%s_%s_%s.dat" % (channel, mass, coupling, var)
                f=open(outdir + card,"w+")
                f.write("file: %sevents_%s_%s_%s_DM.root\n" % (indir,channel,massString[mass],coupling))
                #f.write("name: can_%s\n" % var)
                f.write("name: events\n")
                f.write("var: %s\n" % var)
                f.write("cond: n_l1>0 && n_l2>0\n")
                f.write("logy: 1\n")
                f.write("title: %s GeV\n" % (mass))
                f.write("titley: Events\n")
                f.write("titlex: %s\n" % (titlex[var] if var in titlex else var))
                f.write("minx: %s\n" % (minx[var] if var in minx else "0"))
                f.write("maxx: %s\n" % (maxx[var] if var in maxx else "4000"))
                f.write("miny: 0.5\n")
                f.write("maxy: 50000\n")
                f.write("linecolor: %s\n" % (linecolor[mass] if mass in linecolor else "2"))
                f.write("atlas: Simulation\n")
                f.write("atlasx: 0.6\n")
                f.write("latex: Z'_{DM} \\rightarrow %s\n" % (channellabel[channel]))
                f.write("latexx: 0.73\n")
                f.write("latexy: 0.77\n")
                f.write("sublatex: 10k events\n")
                f.write("sublatexx: 0.73\n")
                f.write("sublatexy: 0.69\n")
                f.close()
