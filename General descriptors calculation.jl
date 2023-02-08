import Pkg
Pkg.add("RDKit")

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
using Plots
y, fs = wavread(raw"C:\Windows\Media\Ring01.wav")

@sk_import ensemble: RandomForestRegressor
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
alc = pyimport("rdkit.Chem.AllChem")


data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI-.csv", DataFrame)
data_minus = (unique(data1,3))[!,[2,3,26]]
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI+.csv", DataFrame)
data_plus = (unique(data2,3))[!,[2,3,26]]
data3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI-.csv", DataFrame)
data_M2_minus = data3[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data4 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI+.csv", DataFrame)
data_M2_plus = data4[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]

histogram(data_M2_minus[:,3],bins=100,xlims=(-3,4.5))
histogram!(data_minus[:,3],bins=100,c=:orange,xlims=(-3,4.5))

## Fingerprint calculation (function)##
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
    desc_full = hcat(rep[:,:],desc_p)
    return desc_full
end

## Fingerprint calculation (calc)##
desc_minus_12 = padel_desc(data_M2_minus[:,:])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\padel_minus_12.csv", desc_minus_12)
desc_plus_12 = padel_desc(data_plus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\padel_plus_12.csv", desc_plus_12)
wavplay(y, fs)

## Morgan FP ##    
function morgan(rep)
    results = pcp.get_compounds(rep[1,1], "name")[1]
    m1 = alc.MolFromSmiles(results.isomeric_smiles)
    fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
    fp_vector = (ones(1024).*1821)'
    for i = 1:1024
        fp_vector[i] = fp[i]
    end
    for j = 2:size(rep,1)
        results = pcp.get_compounds(rep[j,1], "name")[1]
        m1 = alc.MolFromSmiles(results.isomeric_smiles)
        fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
        fp_vector_temp = (ones(1024).*1821)'
        for i = 1:1024
            fp_vector_temp[i] = fp[i]
        end
        fp_vector = vcat(fp_vector,fp_vector_temp)
        println("End of $j compound")
    end    
return fp_vector
end
morgan_minus = morgan(data_minus)
morgan_plus = morgan(data_plus)

morgan_minus_ = DataFrame(hcat(data_minus[:,1], morgan_minus), :auto)
morgan_plus_ = DataFrame(hcat(data_plus[:,1], morgan_plus), :auto)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\morgan_minus.csv", morgan_minus_)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\morgan_plus.csv", morgan_plus_)
