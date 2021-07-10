#include <vector>

// ROOT include(s):
#include <TChain.h>
#include <TString.h>
#include <TCanvas.h>
#include <TError.h>
#include <TH1D.h>
#include <TNtupleD.h>
#include <TFile.h>

// xAOD include(s):
#include "xAODRootAccess/Init.h"
#include "xAODRootAccess/TEvent.h"
#include "xAODRootAccess/tools/ReturnCheck.h"
#include "xAODRootAccess/tools/Message.h"

// EDM include(s):
#include "xAODEgamma/ElectronContainer.h"
#include "xAODTruth/TruthParticleContainer.h"
#include "xAODTruth/TruthEventContainer.h"
#include "xAODTruth/TruthParticle.h"

using namespace std;

void RenormalizeHistByBinWidth(TH1D* hist, bool undo);
TCanvas* plotter(TNtupleD* tuple, TString var, TString cond = "", double min=999, double max=-999);

int main(int argc, char* argv[])
//int main()
{

   // The application's name:
   static const char* APP_NAME = "MC15Validation";

   // Initialise the environment for xAOD reading:
   RETURN_CHECK( APP_NAME, xAOD::Init( APP_NAME ) );

   TString file         = argc > 1 ? string(argv[1]) : "DAOD_TRUTH0.TRUTH.root";
   TString saveFileName = argc > 7 ? string(argv[7]) + "/events.root" : "events.root";

   // Create a TChain with the input file(s):
   TChain chain( "CollectionTree" );
   chain.Add( TString(file) );

   // Create a TEvent object with this TChain as input:
   //xAOD::TEvent event;
   xAOD::TEvent event( xAOD::TEvent::kBranchAccess );
   RETURN_CHECK( APP_NAME, event.readFrom( &chain ) );

   vector<TString> eventVec = {"n_l1","pdgid_l1","status_l1","e_l1","px_l1","py_l1","pz_l1","pt_l1","eta_l1","phi_l1","n_l2","pdgid_l2","status_l2","e_l2","px_l2","py_l2","pz_l2","pt_l2","eta_l2","phi_l2","m_ll"};

   if(file.Contains("DM") && argc >= 6)
   {
       eventVec.push_back("mass");
       eventVec.push_back("width");
       eventVec.push_back("gl");
       eventVec.push_back("xsec");
       eventVec.push_back("massDM");
   }

   TString eventIndex = "";
   for(int i=0; i<eventVec.size(); i++)
   {
       if(i>0) eventIndex += ":";
       eventIndex += eventVec[i];
   }
   TNtupleD* events = new TNtupleD("events","events",eventIndex);

   Double_t GeV = 0.001;


      Double_t n_l1=0, pdgid_l1=0, e_l1=0, px_l1=0, py_l1=0, pz_l1=0, pt_l1=0, eta_l1=0, phi_l1=0;
      Double_t n_l2=0, pdgid_l2=0, e_l2=0, px_l2=0, py_l2=0, pz_l2=0, pt_l2=0, eta_l2=0, phi_l2=0;
      Double_t m_ll=0;
      Int_t status_l1=0, status_l2=0;

   // Loop over the input file(s):
   const ::Long64_t entries = event.getEntries();
   for( ::Long64_t entry = 0; entry < entries; ++entry ) {

      // Load the event:
      if( event.getEntry( entry ) < 0 ) {
         Error( APP_NAME, XAOD_MESSAGE( "Failed to load entry %i" ),
                static_cast< int >( entry ) );
         return 1;
      }

      // Load the truth particles from it:
      const xAOD::TruthParticleContainer* mc_particle = 0;
      RETURN_CHECK( APP_NAME, event.retrieve( mc_particle, "TruthParticles" ) );
      if (entry%1000 == 0) Info( APP_NAME, "Event Number: %i", static_cast< int >( entry ));

      n_l1=0; pdgid_l1=-999; status_l1=-999; e_l1=-999; px_l1=-999; py_l1=-999; pz_l1=-999; pt_l1=-999; eta_l1=-999; phi_l1=-999;
      n_l2=0; pdgid_l2=-999; status_l2=-999; e_l2=-999; px_l2=-999; py_l2=-999; pz_l2=-999; pt_l2=-999; eta_l2=-999; phi_l2=-999;
      m_ll=-999;

//      std::cout << "EVENT " << entry << std::endl;

      //for(auto truth = mc_particle->begin(); truth!=mc_particle->end(); ++truth)
      for(auto truth : *mc_particle)
      {
	

        bool isBSM = ((*truth).pdgId()==5000001);
        bool hasTwoChildren = ((*truth).nChildren()==2);

        bool isDaughter = (*truth).parent() ? ((*truth).parent()->pdgId()==5000001) : false;
        bool isPair = (*truth).parent() ? ((*truth).parent()->nChildren()==2) : false;

        bool isMuon =         (*truth).pdgId()==13;
        bool isAntiMuon =     (*truth).pdgId()==-13;
        bool isElectron =     (*truth).pdgId()==11;
        bool isAntiElectron = (*truth).pdgId()==-11;

        bool isOutgoing =  ((*truth).status()==23);
        bool isHigherPt1 = ((*truth).pt()*GeV > pt_l1);
        bool isHigherPt2 = ((*truth).pt()*GeV > pt_l2);

        if(isBSM)
        {
            auto lep1 = (*truth).child(0);

            n_l1=1;
            pdgid_l1=(*lep1).pdgId();
            status_l1=(*lep1).status();
            e_l1=(*lep1).e()*GeV;
            px_l1=(*lep1).px()*GeV;
            py_l1=(*lep1).py()*GeV;
            pz_l1=(*lep1).pz()*GeV,
            pt_l1=(*lep1).pt()*GeV;
            eta_l1=(*lep1).eta();
            phi_l1=(*lep1).phi();

            if(hasTwoChildren)
            {
              auto lep2 = (*truth).child(1);

              n_l2=1;
              pdgid_l2=(*lep2).pdgId();
              status_l2=(*lep2).status();
              e_l2=(*lep2).e()*GeV;
              px_l2=(*lep2).px()*GeV;
              py_l2=(*lep2).py()*GeV;
              pz_l2=(*lep2).pz()*GeV,
              pt_l2=(*lep2).pt()*GeV;
              eta_l2=(*lep2).eta();
              phi_l2=(*lep2).phi();

              break;
            }
        }
        else if((isMuon or isElectron) && (isOutgoing || isHigherPt1) && (status_l1!=23) )
        {
            n_l1++;
            pdgid_l1=(*truth).pdgId();
            status_l1=(*truth).status();
            e_l1=(*truth).e()*GeV;
            px_l1=(*truth).px()*GeV;
            py_l1=(*truth).py()*GeV;
            pz_l1=(*truth).pz()*GeV,
            pt_l1=(*truth).pt()*GeV;
            eta_l1=(*truth).eta();
            phi_l1=(*truth).phi();
        }
        else if((isAntiMuon or isAntiElectron) && (isOutgoing || isHigherPt2) && (status_l2!=23) )
        {
            n_l2++;
            pdgid_l2=(*truth).pdgId();
            status_l2=(*truth).status();
            e_l2=(*truth).e()*GeV;
            px_l2=(*truth).px()*GeV;
            py_l2=(*truth).py()*GeV;
            pz_l2=(*truth).pz()*GeV,
            pt_l2=(*truth).pt()*GeV;
            eta_l2=(*truth).eta();
            phi_l2=(*truth).phi();
        }

      }

   m_ll=sqrt(fabs(2*pt_l1*pt_l2*(cosh(eta_l1-eta_l2)-cos(phi_l1-phi_l2))));

   vector<double> fill{n_l1,pdgid_l1,status_l1*1.,e_l1,px_l1,py_l1,pz_l1,pt_l1,eta_l1,phi_l1,n_l2,pdgid_l2,status_l2*1.,e_l2,px_l2,py_l2,pz_l2,pt_l2,eta_l2,phi_l2,m_ll};

   if(file.Contains("DM") && argc >= 6)
   {
       fill.push_back((TString(argv[2]).ReplaceAll("p",".")).Atof());
       fill.push_back((TString(argv[3]).ReplaceAll("p",".")).Atof());
       fill.push_back((TString(argv[4]).ReplaceAll("p",".")).Atof());
       fill.push_back((TString(argv[5]).ReplaceAll("p",".")).Atof());
       fill.push_back((TString(argv[6]).ReplaceAll("p",".")).Atof());
   }

   events->Fill(&fill[0]);

   }

   TFile *saveFile = new TFile(saveFileName,"RECREATE");
   events->Write();

   /*
   TCanvas* can_var = 0;

   for(int i=0; i<eventVec.size(); i++)
   {
       can_var = plotter(events,eventVec[i]);
       can_var->Write();
   }
   */

   saveFile->Close();

   // Return gracefully:
   return 0;
}

void RenormalizeHistByBinWidth(TH1D* hist, bool undo)
{
    vector<double> binEdges;
    vector<double> binCenters;
    vector<double> weights;
    vector<double> errors;

    for (int i=1; i <= hist->GetXaxis()->GetNbins(); i++)
    {
        binEdges.push_back(hist->GetXaxis()->GetBinLowEdge(i));
        binCenters.push_back(hist->GetXaxis()->GetBinCenter(i));
        if(undo)
        {
            weights.push_back( hist->GetBinContent(i)*hist->GetXaxis()->GetBinWidth(i) );
            errors.push_back(hist->GetBinError(i-1)*hist->GetXaxis()->GetBinWidth(i) );
        }
        else
        {
            weights.push_back( hist->GetBinContent(i) / hist->GetXaxis()->GetBinWidth(i) );
            errors.push_back(hist->GetBinError(i-1) / hist->GetXaxis()->GetBinWidth(i) );
        }
    }

    binEdges.push_back(hist->GetXaxis()->GetBinUpEdge(hist->GetXaxis()->GetNbins()));
    hist->Reset();
    hist->FillN(binCenters.size(),&binCenters[0],&weights[0]);
    hist->SetError(&errors[0]);
}

TCanvas* plotter(TNtupleD* tuple, TString var, TString cond, double min, double max)
{
    TCanvas* can = new TCanvas("can_"+var,"can_"+var,800,600);

    int n = tuple->Draw(var, cond, "goff");
    if(n==0) cout << "empty!" << endl;

    TH1D* hist1 = (TH1D*)tuple->GetHistogram();
    hist1->SetName(var);
    hist1->GetXaxis()->SetTitle(var);
    //RenormalizeHistByBinWidth(hist1,false);
    if(min<max) hist1->GetXaxis()->SetLimits(min,max);
    hist1->Draw("h");

    return can;
}
