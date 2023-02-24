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
M2M4_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_new.csv", DataFrame)
M2M4_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_new.csv", DataFrame)
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Amide_CNLs.csv", DataFrame)
norman_raw_0 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs_0.csv", DataFrame)

rep = M2M4_minus
sids = pcp.get_sids(rep[1,1], "name")
results = pcp.Compound.from_cid(sids[1]["CID"])
inchi = DataFrame(INCHIKEY=[results.inchikey])

unique(amide_raw,4)

intersect(amide_raw[:,:INCHIKEY],M2M4_plus[:,:INCHIKEY])
intersect(amide_raw[:,:INCHIKEY],M2M4_minus[:,:INCHIKEY])

function inchi_fromname(rep)
    sids = pcp.get_sids(rep[1,1], "name")
    results = pcp.Compound.from_cid(sids[1]["CID"])
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    joined = hcat(DataFrame(rep[1,:]), desc_p)
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            sids = pcp.get_sids(rep[i,1], "name")
            results = pcp.Compound.from_cid(sids[1]["CID"])
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            joined_p = hcat(DataFrame(rep[i,:]), desc_p_temp)
            joined = append!(joined,joined_p)
            println(i)
        catch
            continue
        end
    end
    return joined
end


## Fingerprint calculation (calc)##
fp_minus_12_name = padel_fromname(data_minus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnamecid_new(2).csv",fp_minus_12_name)
fp_plus_12_1 = padel_fromname(data_M4_plus)
fp_plus_12_2= padel_fromSMILES(data_M2_plus[:,:],data4[:,:SMILES])
fp_plus_12_nameSMILES = vcat(fp_plus_12_1,fp_plus_12_2)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_fromnameSMILES_new(2).csv", fp_plus_12_nameSMILES)









## Padel fingerprints optimization ##
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

