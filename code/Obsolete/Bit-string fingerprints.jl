#Bit-string fingerprints

using ScikitLearn
using BSON
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using PyCall
using Conda
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

## load files ##
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI-.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI+_fixedseparator.csv", DataFrame)
data_minus = (unique(data1,3))[!,[2,3,26]]
data_plus = (unique(data2,3))[!,[2,3,26]]

## ESI- ##
coarseness_r = vcat(collect(100:100:1000), collect(1200:300:3000))

for i = 2:length(coarseness_r)
    coarseness = coarseness_r[i]
    ## Fingerprint calculation ##
    results = pcp.get_compounds(data_minus[1,1], "name")[1]
    m = rdk.MolFromSmiles(results.isomeric_smiles)
    FP = rdk.RDKFingerprint(m, fpSize=coarseness)
    FP1 = [FP[i] for i = 1:coarseness]';
    for j = 2:length(data_minus[:,1])
        results = pcp.get_compounds(data_minus[j,1], "name")[1]
        m = rdk.MolFromSmiles(results.isomeric_smiles)
        FP = rdk.RDKFingerprint(m, fpSize=coarseness)
        FP1_temp = [FP[i] for i = 1:coarseness]';
        FP1 = vcat(FP1, FP1_temp)
        println("Itr $i, Compound $j")
    end
    FP_df = DataFrame(FP1, :auto)
    CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\FP_$(coarseness)_minus.csv", FP_df)
end

## ESI+ ##
coarseness_r = vcat(collect(100:100:1000), collect(1200:300:3000))

for i = 1:length(coarseness_r)
    coarseness = coarseness_r[i]
    ## Fingerprint calculation ##
    results = pcp.get_compounds(data_plus[1,1], "name")[1]
    m = rdk.MolFromSmiles(results.isomeric_smiles)
    FP = rdk.RDKFingerprint(m, fpSize=coarseness)
    FP1 = [FP[i] for i = 1:coarseness]';
    for j = 2:length(data_plus[:,1])
        results = pcp.get_compounds(data_plus[j,1], "name")[1]
        m = rdk.MolFromSmiles(results.isomeric_smiles)
        FP = rdk.RDKFingerprint(m, fpSize=coarseness)
        FP1_temp = [FP[i] for i = 1:coarseness]';
        FP1 = vcat(FP1, FP1_temp)
        println("Itr $i, Compound $j")
    end
    FP_df = DataFrame(FP1, :auto)
    CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\FP_$(coarseness)_plus.csv", FP_df)
end
