import Pkg
Pkg.add("Conda")

## import packages ##
using ScikitLearn
using BSON
using LinearAlgebra
using Statistics
using DataFrames
using Plots
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
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI+_fixedseparator.csv", DataFrame)
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

## Parameter optimization ##
coarseness_r = collect(100:100:5000)
leaf_r = collect(4:2:20)
tree_r = vcat(collect(50:50:300),collect(400:100:1000))
MaxFeat = 80   #Int64(ceil(size(desc_temp,2)/3))
itr = 5

z = zeros(itr,6)
for i = 1:itr
    leaf = rand(leaf_r)
    tree = rand(tree_r)
    coarseness = rand(coarseness_r)
    state = rand(1:10)

    ## Fingerprint calculation ##
    results = pcp.get_compounds(data_minus[1,1], "name")[1]
    m = rdk.MolFromSmiles(results.isomeric_smiles)
    FP = rdk.RDKFingerprint(m, fpSize=coarseness)
    FP1 = [FP[i] for i = 1:coarseness]';
for j = 2:50          #length(data_minus[:,1])
    results = pcp.get_compounds(data_minus[j,1], "name")[1]
    m = rdk.MolFromSmiles(results.isomeric_smiles)
    FP = rdk.RDKFingerprint(m, fpSize=coarseness)
    FP1_temp = [FP[i] for i = 1:coarseness]';
    FP1 = vcat(FP1, FP1_temp)
    println("Itr $i, Compound $j")
end


## Parameter optimization ##
coarseness_r = vcat(collect(100:100:1000), collect(1200:300:3000))
leaf_r = collect(4:2:12)
tree_r = vcat(collect(50:50:300),collect(400:100:1000))
itr = 50
z = zeros(itr*length(coarseness_r),6)
for i = 1:length(coarseness_r)
    coarseness = coarseness_r[i]
    FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\FP_plus_$coarseness.csv", DataFrame))
    for j = 1:itr
        leaf = rand(leaf_r)
        tree = rand(tree_r)
        state = rand(1:3)
        MaxFeat = Int64(ceil(coarseness/3))

    ## Regression ##
        X_train, X_test, y_train, y_test = train_test_split(FP1, data_plus[1:end,3], test_size=0.20, random_state=state);
        reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
        fit!(reg, X_train, y_train)
        z[((i-1)*itr+j),1] = leaf
        z[((i-1)*itr+j),2] = tree
        z[((i-1)*itr+j),3] = coarseness
        z[((i-1)*itr+j),4] = state
        z[((i-1)*itr+j),5] = score(reg, X_train, y_train)
        z[((i-1)*itr+j),6] = score(reg, X_test, y_test)
    end
    println("End of $coarseness coarseness")
end    

z_df = DataFrame(leaves = z[:,1], trees = z[:,2], coarseness = z[:,3], state=z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
z_df_sorted_plus = sort(z_df, :accuracy_test, rev=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\z_df_sorted_plus", z_df_sorted_plus)

scatter(z_df_sorted_plus[:,:coarseness],z_df_sorted_plus[:,:accuracy_test],ylims=(0.4,0.5))