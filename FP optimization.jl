import Pkg
Pkg.add("Plots")

## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda

@sk_import ensemble: RandomForestRegressor
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

## load files ##
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI-.csv", DataFrame)
data_minus = (unique(data1,3))[!,[2,3,26]]
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


# Finding compounds with multiple Pubchem CIDs
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
odd_comps = data_minus[odds,:]

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


j=1             ###########

results = pcp.get_compounds(odd_comps[j,1], "name")[1]
m = rdk.MolFromSmiles(results.isomeric_smiles)
FP = rdk.RDKFingerprint(m, fpSize=1048)
FP1 = [FP[i] for i = 1:1048]';
for k=2:(len[odds])[j]
    results = pcp.get_compounds(odd_comps[j,1], "name")[k]
    m = rdk.MolFromSmiles(results.isomeric_smiles)
    FP = rdk.RDKFingerprint(m, fpSize=1048)
    FP1_temp = [FP[i] for i = 1:1048]';
    FP1 = vcat(FP1, FP1_temp)
    println("FP $k")
end

## Parameter optimization ##
coarseness_r = vcat(collect(100:100:1000), collect(1200:300:3000))
leaf_r = collect(4:2:12)
tree_r = vcat(collect(50:50:300),collect(400:100:1000))
itr = 50
z = zeros(itr*length(coarseness_r),6)
for i = 1:length(coarseness_r)
    coarseness = coarseness_r[i]
    FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\FP_minus_$coarseness.csv", DataFrame))
    for j = 1:itr
        leaf = rand(leaf_r)
        tree = rand(tree_r)
        state = rand(1:3)
        MaxFeat = Int64(ceil(coarseness/3))

    ## Regression ##
        X_train, X_test, y_train, y_test = train_test_split(FP1, data_minus[1:end,3], test_size=0.20, random_state=state);
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
z_df_sorted_minus = sort(z_df, :accuracy_test, rev=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\z_df_sorted_minus", z_df_sorted_minus)

scatter(z_df_sorted_minus[:,:coarseness],z_df_sorted_minus[:,:accuracy_test], ylims =(0.5,0.6))


## General descriptors optimization ##
function optim(output, ESI, iterations=50)
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(50:50:300),collect(400:100:1000))
    itr = iterations
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    z = zeros(itr*13,6)
    for i = 0:12
        FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_$(ESI_name)_$i.csv", DataFrame))[:,2:end]
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
        ## Regression ##
            X_train, X_test, y_train, y_test = train_test_split(FP1, output, test_size=0.20, random_state=state);
            reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
            fit!(reg, X_train, y_train)
            z[(i*itr+j),1] = leaf
            z[(i*itr+j),2] = tree
            z[(i*itr+j),3] = i
            z[(i*itr+j),4] = state
            z[(i*itr+j),5] = score(reg, X_train, y_train)
            z[(i*itr+j),6] = score(reg, X_test, y_test)
        end
        println("End of $i FP type (see descriptors.xml)")
    end    
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], FP_type = z[:,3], state=z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

z_df_sorted_plus = optim(data_plus[:,3],+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_plus.csv", z_df_sorted_plus)

opt_res_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_minus.csv", DataFrame)
opt_res_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_plus.csv", DataFrame)

scatter(opt_res_minus[:,3], opt_res_minus[:,end])

## Morgan fingerprints optimization ##
function optim_morgan(output, ESI, iterations=50)
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(50:50:300),collect(400:100:1000))
    itr = iterations
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    z = zeros(itr,5)
        FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\morgan_$(ESI_name).csv", DataFrame))[:,2:end]
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
        ## Regression ##
            X_train, X_test, y_train, y_test = train_test_split(FP1, output, test_size=0.20, random_state=state);
            reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = score(reg, X_train, y_train)
            z[j,5] = score(reg, X_test, y_test)
            println("End of $j / $itr iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], accuracy_train = z[:,4], accuracy_test = z[:,5])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

z_df_sorted_minus = optim_morgan(data_minus[:,3],-1)
z_df_sorted_plus = optim_morgan(data_plus[:,3],+1)

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Morgan_FP_optimisation_results_minus.csv", z_df_sorted_minus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Morgan_FP_optimisation_results_plus.csv", z_df_sorted_plus)

## Meeting notes ##
#  Morgan Fingerprints (if time) --> DONE
## Importance
## Inter-Correlation
## Positive mode
# Residuals + plots
## Find abnormalities to show model strengths and weaknesses
## Different types of models (CATBOOST, XgBoost)
# CATBOOST -> Learning rate, LU1_reg, LU2_reg
# XgBoost -> Learning rate & RandomForestRegressor

## CNL ##
## InChI keys creation
## Talk with Denice


