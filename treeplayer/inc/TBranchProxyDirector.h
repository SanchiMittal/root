// @(#)root/base:$Name:  $:$Id: TBranchProxyDirector.h,v 1.3 2004/06/28 16:42:12 brun Exp $
// Author: Philippe Canal 13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxyDirector
#define ROOT_TBranchProxyDirector

#ifndef ROOT_TTree
#include "TTree.h"
#endif
#include <list>
#ifdef R__HPUX
namespace std {
   using ::list;
}
#endif

namespace ROOT {
   class TBranchProxy;
   class TFriendProxy;

   class TBranchProxyDirector {

      //This class could actually be the selector itself.
      TTree *fTree;
      Long64_t fEntry;
      std::list<TBranchProxy*> fDirected;
      std::list<TFriendProxy*> fFriends;

   public:

      TBranchProxyDirector(TTree* tree, Long64_t i);
      TBranchProxyDirector(TTree* tree, Int_t i);     // cint has (had?) a problem casting int to long long

      void     Attach(TBranchProxy* p);
      void     Attach(TFriendProxy* f);
      Long64_t GetReadEntry() const;
      TTree*   GetTree() const;
      // void   Print();
      void     SetReadEntry(Long64_t entry);
      TTree*   SetTree(TTree *newtree);

   };

} /* namespace ROOT */

#endif
