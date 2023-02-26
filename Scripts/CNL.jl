## import packages ##
using ScikitLearn
using BSON
using Plots
#using Statistics
using DataFrames
using CSV
#using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda

@sk_import ensemble: RandomForestRegressor
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

## load files ##
M2M4_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_w_inchikey.csv", DataFrame)
M2M4_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_w_inchikey.csv", DataFrame)
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Amide_CNLs.csv", DataFrame)
norman_raw_0 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs_0.csv", DataFrame)
norman_raw_1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs_1.csv", DataFrame))
norman_raw_2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs_2.csv", DataFrame)

norman_raw = append!(append!(norman_raw_0,norman_raw_1),norman_raw_2);
# Find differences in M2M4_minus with and without inchikey
vec1 = M2M4_minus[:,1]
vec2 = M2M4_minus_experim[:,1]
i = 1
while i <= min(length(vec1), length(vec2))
    if vec1[i] != vec2[i]
        println("First non-matching element found at index ", i)
        break
    end
    i += 1
end
M2M4_minus[1210,:]
M2M4_minus_experim[1210,:]      # Here is the difference

# Extract the CNL from the raw files (Note: not all of these are CNLs)

CNL_amide = amide_raw[1:5,8:100008]
CNL_norman_0 = norman_raw_0[1:5,9:100009]
sulfa = norman_raw_0[norman_raw_0[:,2] .== "Sulfamoxole",:]
sulfa[1,:] .== sulfa[2,:]

# Find similar elements in CNL dataset and IE dataset to create a CNL-IE training set for the models
M2M4_inchikeys = vcat(M2M4_plus[:,:INCHIKEY],M2M4_minus[:,:INCHIKEY])

v1 = intersect(Vector(amide_raw[:,:INCHIKEY]),M2M4_inchikeys)
v2 = intersect(Vector(norman_raw_0[:,:INCHIKEY]),M2M4_inchikeys)
v3 = intersect(Vector(norman_raw_1[:,:INCHIKEY]),M2M4_inchikeys)
v4 = intersect(Vector(norman_raw_2[:,:INCHIKEY]),M2M4_inchikeys)
#common_inchikeys = vcat(vcat(v1,v2), vcat(v3,v4))
common_inchikeys = vcat(v1,v2)

unique(amide_raw, :NAME)
unique(norman_raw_0, :NAME)

for i = 1:length(common_inchikeys)
    key = common_inchikeys[i]
    amide = findall(x -> x .== key, amide_raw)
    findall(x -> x .== key, amide_raw)
    findall(x -> x .== key, amide_raw)
    findall(x -> x .== key, amide_raw)


end














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

