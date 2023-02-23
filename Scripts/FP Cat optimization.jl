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
cat = pyimport("catboost")
## load files ##
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI-.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI+.csv", DataFrame)
data3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI-.csv", DataFrame)
data4 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI+.csv", DataFrame)

data_M4_minus = (unique(data1,2))[!,[2,3,27,28,29,30,31]]
data_M4_plus = (unique(data2,2))[!,[2,3,27,28,29,30,31]]
data_M2_minus = data3[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_M2_plus = data4[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_minus = vcat(data_M4_minus, data_M2_minus)
data_plus = vcat(data_M4_plus, data_M2_plus)


## Type-12 fingerprints optimization ##
function optim_type12(ESI, iterations=50)
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    learn_r = collect(0.01:0.01:0.15)
    itr = iterations
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    z = zeros(itr,6)
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_$(ESI_name)_12_new.csv", DataFrame)
        FP1 = Matrix(hcat(FP[!,:pH_aq],FP[!,8:end]))
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            learn_rate = rand(learn_r)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
        ## Regression ##
            X_train, X_test, y_train, y_test = train_test_split(FP1, FP[!,:logIE], test_size=0.20, random_state=state);
            reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=leaf,verbose=false)
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = learn_rate
            z[j,5] = score(reg, X_train, y_train)
            z[j,6] = score(reg, X_test, y_test)
            println("End of $j / $itr iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], learn_rate = z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end


z = optim_type12(-1,50)     # Default
z = optim_type12(-1,49)     # Lossguide, min_samples_leaf -> default
z = optim_type12(-1,48)     # Lossguide, min_samples_leaf -> 4:10
scatter(z[:,:learn_rate], z[:,:accuracy_test],ylims=(0.7,0.8),xlims=(0,0.2))

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Morgan_FP_optimisation_results_plus.csv", z_df_sorted_plus)