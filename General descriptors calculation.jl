# General descriptors calculation
using ScikitLearn
using BSON
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
using WAV
y, fs = wavread(raw"C:\Windows\Media\Ring01.wav")

@sk_import ensemble: RandomForestRegressor
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")


data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI-.csv", DataFrame)
data_minus = (unique(data1,3))[!,[2,3,26]]
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI+_fixedseparator.csv", DataFrame)
data_plus = (unique(data2,3))[!,[2,3,26]]

## Fingerprint calculation ##

function padel_desc(rep)
    results = pcp.get_compounds(rep[1,1], "name")[1]
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            results = pcp.get_compounds(rep[i,1], "name")[1]
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            desc_p = vcat(desc_p,desc_p_temp)
            println(i)
        catch
            continue
        end
    end
    desc_full = hcat(rep[:,1],desc_p)
    return desc_full
end


desc_minus_12 = padel_desc(data_minus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\padel_minus_12.csv", desc_minus_12)
wavplay(y, fs)


desc_plus_12 = padel_desc(data_plus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\padel_plus_12.csv", desc_plus_12)
wavplay(y, fs)