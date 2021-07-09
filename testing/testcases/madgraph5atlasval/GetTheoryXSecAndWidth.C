#include "TNtupleD.h"
#include "TFile.h"
#include "TParameter.h"
#include <iostream>

void GetTheoryXSecAndWidth(const char* inputFileName, const char* outputFileName) {

    TFile inFile(inputFileName, "READ");
    TNtupleD *tree = (TNtupleD*)inFile.Get("events");
    double xsec(0.0), width(0.0);
    tree->SetBranchAddress("xsec", &xsec);
    tree->SetBranchAddress("width", &width);
    double acceptance = (double)tree->Draw("mass", "(pt_l1>30 && pt_l2>30 && TMath::Abs(eta_l1)<2.5 && TMath::Abs(eta_l2)<2.5 && m_ll>((1000*mass)-(2*width)))");
    acceptance /= (double)tree->GetEntries();
    tree->GetEntry(0);

    TParameter<double> fidXsecForFile("fidXS", xsec * acceptance);
    TParameter<double> widthForFile("width", width);
    
    TFile outFile(outputFileName, "RECREATE");
    fidXsecForFile.Write();
    widthForFile.Write();
    outFile.Close();

    cout << "Saved width and fiducial cross-section in file "
	 << outFile.GetName() << endl;
    
}
