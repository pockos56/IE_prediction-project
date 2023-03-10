## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
#using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda

## load files ##
M2M4_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_w_inchikey.csv", DataFrame)
# What is going on with this? # M2M4_minus_2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_w_inchikey2.csv", DataFrame)
M2M4_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_w_inchikey.csv", DataFrame)
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Amide_CNLs.csv", DataFrame)
norman_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs.csv", DataFrame)

# Preprocessing
norman_original = norman_raw        #Keeping the original Norman dataset
select!(norman_raw, Not([:leverage, :Pred_RTI_Pos_ESI, :Pred_RTI_Neg_ESI])) #Changing the norman dataset to the Amide format


# Find differences in M2M4_minus with and without inchikey
vec1 = M2M4_minus[:,1]
vec2 = M2M4_minus_2[:,1]
i = 1
while i <= min(length(vec1), length(vec2))
    if vec1[i] != vec2[i]
        println("First non-matching element found at index ", i)
        break
    end
    i += 1
end
M2M4_minus_2[1210,:]      # Here is the difference

# Extract the CNL from the raw files (Note: not all of these are CNLs)

#CNL_amide = amide_raw[:,8:100008]
#CNL_norman = norman_raw[:,9:100009]
sulfa = norman_raw_0[norman_raw_0[:,2] .== "Sulfamoxole",:]
sulfa[1,:] .== sulfa[2,:]

# Find similar elements in CNL dataset and IE dataset to create a CNL-IE training set for the models
M2M4_names = unique(vcat(M2M4_plus[:,:name],M2M4_minus[:,:name]))       # 1086 names
M2M4_inchikeys = unique(vcat(M2M4_plus[:,:INCHIKEY],M2M4_minus[:,:INCHIKEY]))       # 1035 Inchikeys
unique(M2M4_minus,1)            # 335 
unique(M2M4_minus,8)            # 331
unique(M2M4_plus,1)             # 880
unique(M2M4_plus,8)             # 841

v1_inchi = intersect(Vector(amide_raw[:,:INCHIKEY]),M2M4_inchikeys)     #25
v1_name = intersect(Vector(amide_raw[:,:NAME]),M2M4_names)              #11
v2_inchi = intersect(Vector(norman_raw[:,:INCHIKEY]),M2M4_inchikeys)
v2_name = intersect(Vector(norman_raw[:,:NAME]),M2M4_names)
common_inchikeys = unique(vcat(v1_inchi,v2_inchi))                      # Common InChiKeys between the CNL and IE datasets

unique(amide_raw, :NAME)                # 133 compounds in the amide dataset
unique(amide_raw, :INCHIKEY)                # 133 compounds in the amide dataset

unique(norman_raw, :INCHIKEY)               # 3217 compounds in the NORMAN dataset

function createCNLIEdataset(mode=:sum)
    #z = zeros(length(common_inchikeys),size(amide_raw,2))
    if mode != :sum
        error("Please use :sum or ... for the mode argument!")
    end

    for i = 1:length(common_inchikeys)
        key = common_inchikeys[i]
        amide = amide_raw[amide_raw[:,:INCHIKEY] .== key, :]
        norman = norman_raw[norman_raw[:,:INCHIKEY] .== key,:]
        CNL_set_temp = vcat(amide,norman)
        if size(CNL_set_temp,1) == 0
            error("Compound with INCHIKEY at index $i has no match")
        elseif size(CNL_set_temp,1) == 1
            z = CNL_set_temp[:,1:100008]
        elseif size(CNL_set_temp,1) > 1
            if mode == :sum
                cnl_first_row = CNL_set_temp[1,1:100008]
                cnl_only = CNL_set_temp[:,8:100008]
                sum_row = sum(Matrix(cnl_only), dims=1)
                cnl_first_row[8:end] = sum_row
                z = cnl_first_row
            end
        end

        # Finding the IE in the IE dataset
        minus = M2M4_minus[M2M4_minus[:,:INCHIKEY] .== key, :]
        plus = M2M4_plus[M2M4_plus[:,:INCHIKEY] .== key, :]
        IE_set_temp = vcat(minus,plus)

        z[3] = 


        # Recording
        if i == 1           

        elseif i > 1

            if i ==length(common_inchikeys)
                rename!(z, :RI_pred => :logIE)
            end
        end
        return z
    end
end


target = 3
sum(sum(Matrix(cnl_only[:,8:end]) .== target, dims=1))

# 94 ones, 1241758 minus ones, 













## ??Padel fingerprints optimization ##
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

# We need to keep every entry as is, without combining.
