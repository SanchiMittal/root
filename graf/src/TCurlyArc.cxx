// @(#)root/graf:$Name$:$Id$
// Author: Otto Schaile   20/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// This class implements curly or wavy arcs typically used to draw Feynman diagrams.
// Amplitudes and wavelengths may be specified in the constructors,
// via commands or interactively from popup menus.
// The class make use of TCurlyLine by inheritance, ExecuteEvent methods
// are highly inspired from the methods used in TPolyLine and TArc.
// The picture below has been generated by the tutorial feynman.
//Begin_Html
/*
<img src="gif/feynman.gif">
*/
//End_Html
//________________________________________________________________________

#include "iostream.h"
#include "TCurlyArc.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TMath.h"

ClassImp(TCurlyArc)

//_____________________________________________________________________________________
TCurlyArc::TCurlyArc(Float_t x1, Float_t y1,
                   Float_t rad, Float_t phimin, Float_t phimax,
                   Float_t tl, Float_t trad)
         : fR1(rad), fPhimin(phimin),fPhimax(phimax)
{
 // create a new TCurlyarc with center (x1, y1) and radius rad.
 // The wavelength and amplitude are given in percent of the line length
 // phimin and phimax are given in degrees.

   fX1         = x1;
   fY1         = y1;
   fIsCurly    = kTRUE;
   fAmplitude  = trad;
   fWaveLength = tl;
   fTheta      = 0;
   Build();
}

//_____________________________________________________________________________________
void TCurlyArc::Build()
{
//*-*-*-*-*-*-*-*-*-*-*Create a curly (Gluon) or wavy (Gamma) arc*-*-*-*-*-*
//*-*                  ===========================================
   Float_t dang = fPhimax - fPhimin;
   if(dang < 0) dang += 360;
   Float_t length = TMath::Pi() * fR1 * dang/180;
   Float_t x1sav = fX1;
   Float_t y1sav = fY1;
   fX1 = fY1 = 0;
   fX2 = length;
   fY2 = 0;
   TCurlyLine::Build();
   fX1 = x1sav;
   fY1 = y1sav;
   Float_t *xv= GetX();
   Float_t *yv= GetY();
   Float_t xx, yy, angle;
   for(Int_t i = 0; i < fNsteps; i++){
      angle = xv[i] / (fR1) + fPhimin * TMath::Pi()/180;
      xx    = (yv[i] + fR1) * cos(angle);
      yy    = (yv[i] + fR1) * sin(angle);
      xv[i] = xx + fX1;
      yv[i] = yy + fY1;
   }

}

//______________________________________________________________________________
Int_t TCurlyArc::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to an arc*-*-*-*
//*-*                  ===========================================
//  Compute the closest distance of approach from point px,py to this arc.
//  The distance is computed in pixels units.
//

//*-*- Compute distance of point to center of arc
   Int_t pxc    = gPad->XtoAbsPixel(fX1);
   Int_t pyc    = gPad->YtoAbsPixel(fY1);
   Float_t dist = TMath::Sqrt(Float_t((pxc-px)*(pxc-px)+(pyc-py)*(pyc-py)));
   Float_t cosa = (px - pxc)/dist;
   Float_t sina = (pyc - py)/dist;
   Float_t phi  = TMath::ATan2(sina,cosa);
   if(phi < 0) phi += 2 * TMath::Pi();
   phi = phi * 180 / TMath::Pi();
   if(fPhimax > fPhimin){
      if(phi < fPhimin || phi > fPhimax) return 9999;
   } else {
      if(phi > fPhimin && phi < fPhimax) return 9999;
   }
   Int_t pxr = gPad->XtoAbsPixel(fR1 + gPad->GetUxmin());
   Float_t distr = TMath::Abs(dist-pxr);
   return Int_t(distr);
}

//______________________________________________________________________________
void TCurlyArc::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function is called when a TCurlyArc is clicked with the locator
//
//  If Left button clicked on one of the line end points, this point
//     follows the cursor until button is released.
//
//  if Middle button clicked, the line is moved parallel to itself
//     until the button is released.
//

   Int_t kMaxDiff = 10;
   const Int_t np = 10;
   const Float_t PI = 3.141592;
   static Int_t x[np+3], y[np+3];
   static Int_t px1,py1,npe,R1;
   static Int_t pxold, pyold;
   Int_t i, dpx, dpy;
   Float_t angle,dx,dy,dphi,rTy,rBy,rLx,rRx;
   Float_t  phi0;
   static Bool_t T, L, R, B, INSIDE;
   static Int_t Tx,Ty,Lx,Ly,Rx,Ry,Bx,By;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();
      dphi = (fPhimax-fPhimin) * PI / 180;
      if(dphi<0) dphi += 2 * PI;
      dphi /= np;
      phi0 = fPhimin * PI / 180;
       for (i=0;i<=np;i++) {
         angle = Float_t(i)*dphi + phi0;
         dx    = fR1*TMath::Cos(angle);
         dy    = fR1*TMath::Sin(angle);
         x[i]  = gPad->XtoAbsPixel(fX1 + dx);
         y[i]  = gPad->YtoAbsPixel(fY1 + dy);
      }
      if (fPhimax-fPhimin >= 360 ) {
         x[np+1] = x[0];
         y[np+1] = y[0];
         npe = np;
      } else {
         x[np+1]   = gPad->XtoAbsPixel(fX1);
         y[np+1]   = gPad->YtoAbsPixel(fY1);
         x[np+2] = x[0];
         y[np+2] = y[0];
         npe = np + 2;
      }
      px1 = gPad->XtoAbsPixel(fX1);
      py1 = gPad->YtoAbsPixel(fY1);
      Tx = Bx = px1;
      Ly = Ry = py1;
      Ty = gPad->YtoAbsPixel( fR1+fY1);
      By = gPad->YtoAbsPixel(-fR1+fY1);
      Lx = gPad->XtoAbsPixel(-fR1+fX1);
      Rx = gPad->XtoAbsPixel( fR1+fX1);
      R1 = TMath::Abs(By-Ty)/2;
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      // No break !!!

   case kMouseMotion:
      px1 = gPad->XtoAbsPixel(fX1);
      py1 = gPad->YtoAbsPixel(fY1);
      Tx = Bx = px1;
      Ly = Ry = py1;
      Ty = gPad->YtoAbsPixel(fR1+fY1);
      By = gPad->YtoAbsPixel(-fR1+fY1);
      Lx = gPad->XtoAbsPixel(-fR1+fX1);
      Rx = gPad->XtoAbsPixel(fR1+fX1);
      T = L = R = B = INSIDE = kFALSE;
      if ((TMath::Abs(px - Tx) < kMaxDiff) &&
          (TMath::Abs(py - Ty) < kMaxDiff)) {             // top edge
         T = kTRUE;
         gPad->SetCursor(kTopSide);
      }
      else
      if ((TMath::Abs(px - Bx) < kMaxDiff) &&
          (TMath::Abs(py - By) < kMaxDiff)) {             // bottom edge
         B = kTRUE;
         gPad->SetCursor(kBottomSide);
      }
      else
      if ((TMath::Abs(py - Ly) < kMaxDiff) &&
          (TMath::Abs(px - Lx) < kMaxDiff)) {             // left edge
         L = kTRUE;
         gPad->SetCursor(kLeftSide);
      }
      else
      if ((TMath::Abs(py - Ry) < kMaxDiff) &&
          (TMath::Abs(px - Rx) < kMaxDiff)) {             // right edge
         R = kTRUE;
         gPad->SetCursor(kRightSide);
      }
      else {INSIDE= kTRUE; gPad->SetCursor(kMove); }
      pxold = px;  pyold = py;

      break;

   case kButton1Motion:
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      for (i=0;i<npe;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      if (T) {
         R1 -= (py - pyold);
      }
      if (B) {
         R1 += (py - pyold);
      }
      if (L) {
         R1 -= (px - pxold);
      }
      if (R) {
         R1 += (px - pxold);
      }
      if (T || B || L || R) {
         gVirtualX->SetLineColor(-1);
         TAttLine::Modify();
         dphi = (fPhimax-fPhimin) * PI / 180;
         if(dphi<0) dphi += 2 * PI;
         dphi /= np;
         phi0 = fPhimin * PI / 180;
         Float_t uR1 = gPad->PixeltoX(R1) - gPad->GetUxmin();
         Int_t pX1   = gPad->XtoAbsPixel(fX1);
         Int_t pY1   = gPad->YtoAbsPixel(fY1);
         for (i=0;i<=np;i++) {
            angle = Float_t(i)*dphi + phi0;
            dx    = uR1 * TMath::Cos(angle);
            dy    = uR1 * TMath::Sin(angle);
            x[i]  = gPad->XtoAbsPixel(fX1 + dx);
            y[i]  = gPad->YtoAbsPixel(fY1 + dy);
         }
         if (fPhimax-fPhimin >= 360 ) {
            x[np+1] = x[0];
            y[np+1] = y[0];
            npe = np;
         } else {
            x[np+1]   = pX1;
            y[np+1]   = pY1;
            x[np+2] = x[0];
            y[np+2] = y[0];
            npe = np + 2;
         }
         for (i=0;i<npe;i++) {
            gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
         }
      }
      if (INSIDE) {
          dpx  = px-pxold;  dpy = py-pyold;
          px1 += dpx; py1 += dpy;
          for (i=0;i<=npe;i++) { x[i] += dpx; y[i] += dpy;}
          for (i=0;i<npe;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      Tx = Bx = px1;
      Rx = px1+R1;
      Lx = px1-R1;
      Ry = Ly = py1;
      Ty = py1-R1;
      By = py1+R1;
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      pxold = px;
      pyold = py;
      break;

   case kButton1Up:
      fX1 = gPad->AbsPixeltoX(px1);
      fY1 = gPad->AbsPixeltoY(py1);
      rBy = gPad->AbsPixeltoY(py1+R1);
      rTy = gPad->AbsPixeltoY(py1-R1);
      rLx = gPad->AbsPixeltoX(px1+R1);
      rRx = gPad->AbsPixeltoX(px1-R1);
      fR1 = TMath::Abs(rRx-rLx)/2;
      fR1 = TMath::Abs(rTy-rBy)/2;
      Build();
      gPad->Modified(kTRUE);
      gVirtualX->SetLineColor(-1);
   }
}

//_____________________________________________________________________________________
void TCurlyArc::SavePrimitive(ofstream &out, Option_t *){
    // Save primitive as a C++ statement(s) on output stream out

   if (gROOT->ClassSaved(TCurlyArc::Class())) {
       out<<"   ";
   } else {
       out<<"   TCurlyArc *";
   }
   out<<"curlyarc = new TCurlyArc("
     <<fX1<<","<<fY1<<","<<fR1<<","<<fPhimin<<","<<fPhimax<<","
      <<fWaveLength<<","<<fAmplitude<<");"<<endl;
  if (!fIsCurly) {
      out<<"   curlyarc->SetWavy();"<<endl;
   }
   SaveLineAttributes(out,"curlyarc",1,1,1);
   out<<"   curlyarc->Draw();"<<endl;
}


//_____________________________________________________________________________________
void TCurlyArc::SetCenter(Float_t x, Float_t y)
{
   fX1 = x;
   fY1 = y;
   Build();
}
void TCurlyArc::SetRadius(Float_t x)
{
   fR1 = x;
   Build();
}
void TCurlyArc::SetPhimin(Float_t x)
{
   fPhimin = x;
   Build();
}
void TCurlyArc::SetPhimax(Float_t x)
{
   fPhimax = x;
   Build();
}
