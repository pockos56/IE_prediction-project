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
using Random
cat = pyimport("catboost")

function CNL_optim(ESI; iterations=100, split_size=0.2)

    if ESI == -1
        ESI_name = "NEG"
    elseif ESI == 1
        ESI_name = "POS"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    data_whole = vcat(vcat(norman, amide), MB)

    # Set ranges for hyperparameters
    leaf_r = collect(4:1:10)
    tree_r = vcat(collect(20:20:100), vcat(collect(150:50:400),collect(500:100:1000)))
    learn_r = vcat(collect(0.01:0.01:0.15), collect(0.2:0.1:0.8))

        z = zeros(iterations,7)
        variables = Matrix(hcat(data_whole[!,:pH_aq],data_whole[!,8:end]))
        for j = 1:iterations
        # Pick one random hyperparameter
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            random_seed = rand(1:3)
            learn_rate = rand(learn_r)
            MaxFeat = Int64(ceil(size(variables,2)/3))
        # Split
            Random.seed!(random_seed)
            classes = unique(data_whole[:,3])
            test_set_indices = findall(x -> x in classes[rand(1:length(classes),Int(round(split_size * length(classes))))], data_whole[:,:INCHIKEY])
            train_set_indices = setdiff(collect(1:size(data_whole,1)),test_set_indices)
            X_train = variables[train_set_indices,:]
            X_test = variables[test_set_indices,:]
            y_train =  data_whole[train_set_indices,:logIE]
            y_test = data_whole[test_set_indices,:logIE]
            #X_train, X_test, y_train, y_test = train_test_split(variables, data_whole[!,:logIE], test_size=0.20, random_state=state);
            println("$j / $iterations: Train/Test set split")

        ## Regression ##
            reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf,verbose=false)
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = learn_rate
            z[j,5] = score(reg, X_train, y_train)
            z[j,6] = score(reg, X_test, y_test)
            z[j,7] = round(length(test_set_indices)/size(data_whole,1), digits=4)
            println("End of $j / $iterations iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], learn_rate = z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6], actual_split_size = z[:,7])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end
optimization_df_neg = CNL_optim(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_NEG.CSV", optimization_df_neg)
optimization_df_pos = CNL_optim(1, iterations=20)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_POS.CSV", optimization_df_pos)
