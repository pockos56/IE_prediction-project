import Pkg
Pkg.add("Conda")

## import packages ##
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

@sk_import ensemble: RandomForestRegressor
#rdk = pyimport("rdkit")
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
#pd = pyimport("padelpy")



## load files ##
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI-_fixedseparator.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI+_fixedseparator.csv", DataFrame)
data_minus = (unique(data1,3))[!,[2,3,26]]
data_plus = (unique(data2,3))[!,[2,3,26]]



## Fingerprint calculation ##
coarseness = 1048
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
    println(j)
end

#= Finding compounds with multiple Pubchem CIDs
results = pcp.get_compounds(data_minus[1,1], "name")
len = length(results)
for j = 2:length(data_minus[:,1])
    len_temp = -1
    results = pcp.get_compounds(data_minus[j,1], "name")
    len_temp = length(results)
    len = vcat(len, len_temp)
    println(j)
end

odds = findall(x -> x!=1, len)
odd_names = hcat(data_minus[odds,1], len[odds])
=#

## RF regression ##
X_train, X_test, y_train, y_test = train_test_split(FP_saved, data_minus[:,3], test_size=0.20, random_state=21);
coarseness_r = collect(100:100:5000)
leaf_r = collect(4:2:20)
tree_r = vcat(collect(50:50:300),collect(400:100:1000))
MaxFeat = 80   #Int64(ceil(size(desc_temp,2)/3))
itr = 20

z = zeros(itr,5)
for i = 1:itr
    leaf = rand(leaf_r)
    tree = rand(tree_r)
    coarseness = rand(coarseness_r)

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
    println(j)
end




## Regression ##
    reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=21)
    fit!(reg, X_train, y_train)
    z[i,1] = leaf
    z[i,2] = tree
    z[i,3] = coarseness
    z[i,4] = score(reg, X_train, y_train)
    z[i,5] = score(reg, X_test, y_test)
    println(i)
end    

z_df = DataFrame(leaves = z[:,1], trees = z[:,2], accuracy_train = z[:,4], accuracy_test = z[:,5])
z_df_sorted = sort(z_df, :accuracy_test, rev=true)
