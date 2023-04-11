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


function CNL_optim(ESI; iterations::Int=200, split_size::Float64=0.2)
    if ESI == -1
        ESI_name = "NEG"
    elseif ESI == 1
        ESI_name = "POS"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))
    NIST = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))
    data_whole = vcat(vcat(vcat(norman, amide), MB), NIST)


    # Set ranges for hyperparameters
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(20:40:80), vcat(collect(100:100:900),(collect(1000:200:1400))))
    random_seed_r = collect(1:1:3)
    learn_r = vcat(collect(0.01:0.02:0.15), collect(0.2:0.2:0.8))
    l2_leaf_reg_r = vcat(collect(2:2:6),collect(10:5:15))

    z = zeros(iterations,8)
    variables = Matrix(hcat(data_whole[!,:pH_aq],data_whole[!,8:end]))
    for j = 1:iterations
        # Pick one random hyperparameter
        leaf = rand(leaf_r)
        tree = rand(tree_r)
        random_seed = rand(random_seed_r)
        learn_rate = rand(learn_r)
        l2_leaf_reg = rand(l2_leaf_reg_r)
        # New splitting
        classes = unique(data_whole[:,3])
        test_set_inchikeys = classes[randperm(MersenneTwister(random_seed),length(classes))[1:Int(round(split_size*length(classes)))]]
        train_set_inchikeys = classes[randperm(MersenneTwister(random_seed),length(classes))[1+(Int(round(split_size*length(classes)))):end]]
        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

        X_train = variables[train_set_indices,:]
        X_test = variables[test_set_indices,:]
        y_train =  data_whole[train_set_indices,:logIE]
        y_test = data_whole[test_set_indices,:logIE]
        println("$j / $iterations: Train/Test set split")

        ## Regression ##
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf,l2_leaf_reg = l2_leaf_reg, rsm=0.3, verbose=false)
        @time fit!(reg, X_train, y_train)
        score(reg, X_train, y_train)





        

        reg.boosting_type_
        z[j,1] = leaf
        z[j,2] = tree
        z[j,3] = random_seed
        z[j,8] = l2_leaf_reg
        z[j,4] = learn_rate
        z[j,5] = round(score(reg, X_train, y_train), digits=4)
        z[j,6] = round(score(reg, X_test, y_test), digits=4)
        z[j,7] = round(length(test_set_indices)/size(data_whole,1), digits=4)
        println("End of $j / $iterations iterations (ESI $ESI_name)")
    end
    z_df = DataFrame(leaves=z[:,1],trees=z[:,2],l2_leaf_reg=z[:,8],learn_rate=z[:,4],state=z[:,3],accuracy_train=z[:,5],accuracy_test=z[:,6],actual_test_size=z[:,7])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

optimization_df_neg = CNL_optim(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_NEG.CSV", optimization_df_neg)
optimization_df_pos = CNL_optim(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_POS.CSV", optimization_df_pos)
