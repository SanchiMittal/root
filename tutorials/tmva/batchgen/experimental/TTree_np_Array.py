from ROOT import TMVA, TFile, TTree, TCut
import numpy as np
from array import array

data = TFile.Open("Higgs_data.root")
df1 = data.Get("sig_tree")
df2 = data.Get("bkg_tree")

df1.Print()


n = array('f', [ 0. ])
branch_list = df1.GetListOfBranches()

df_mat = []

for branch in df1.GetListOfBranches():
    name = branch.GetName()
    #print(name)
    df_list = []
    for i in range(df1.GetEntries()):
        n = array('f', [ 0. ])
        df1.SetBranchAddress(name, n)
        df1.GetEntry(i)
        df_list.append(n)
        del n
    df_mat.append(df_list)
print(df_mat[:][0])



