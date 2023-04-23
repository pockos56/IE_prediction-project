## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using Random
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
cat = pyimport("catboost")

## Type-12 fingerprints optimization ##
function optim_type12(ESI; itr::Int=100, grow = :Lossguide)
    leaf_r = vcat(collect(4:2:10),15)
    tree_r = vcat(collect(20:20:100), vcat(collect(150:50:400),collect(500:100:1000)))
    learn_r = vcat(collect(0.01:0.02:0.15), collect(0.2:0.1:0.8))
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    function split_classes(FP; random_state::Int=1312, split_size::Float64=0.2)
        classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_FPs = Matrix(FP[indices,9:end])
    
        function leverage_dist(unique_comps_FPs, Norman)
            ZZ = pinv(transpose(unique_comps_FPs) * unique_comps_FPs)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * ZZ * x
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_FPs,unique_comps_FPs)
   
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
        return inchi_train, inchi_test
    end

        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        FP1 = Matrix(hcat(FP[!,:pH_aq],FP[!,9:end]))
        z = zeros(itr,7)

        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            learn_rate = rand(learn_r)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
            X_train, X_test, y_train, y_test = split_classes(FP; random_state=state)

                # Old way  
            # X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=state);
                # New way
            ## Regression ##
            reg = ()
            if grow == :Lossguide
                reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=state, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
            elseif grow == :SymmetricTree
                reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=state, grow_policy=:SymmetricTree, min_data_in_leaf=leaf, verbose=false)
            else error("Provide a valid grow policy (:Lossguide or :SymmetricTree)")
            end
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = learn_rate
            z[j,5] = score(reg, X_train, y_train)
            z[j,6] = score(reg, X_test, y_test)
            z[j,7] = size(X_test,1)/(size(X_train,1)+size(X_test,1))
            println("End of $j / $itr iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], learn_rate = z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

hyperparameter_neg_loss = optim_type12(-1, grow = :Lossguide)     # Default
hyperparameter_neg_sym = optim_type12(-1, grow = :SymmetricTree)     
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_Cat_hyperparameter_neg_loss.csv", hyperparameter_neg_loss)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_Cat_hyperparameter_neg_sym.csv", hyperparameter_neg_sym)

hyperparameter_pos_loss = optim_type12(+1, grow = :Lossguide)     # Default
hyperparameter_pos_sym = optim_type12(+1, grow = :SymmetricTree)   
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_Cat_hyperparameter_pos_loss.csv", hyperparameter_pos_loss)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_Cat_hyperparameter_pos_sym.csv", hyperparameter_pos_sym)


## Other fingeprints optimization ##
function optim_type12(ESI; iterations::Int=100, grow = :Lossguide)
    leaf_r = vcat(collect(4:2:10),15)
    tree_r = vcat(collect(20:20:100), vcat(collect(150:50:400),collect(500:100:1000)))
    learn_r = vcat(collect(0.01:0.02:0.15), collect(0.2:0.1:0.8))
    itr = iterations
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    function split_classes(ESI; random_state::Int=1312)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_fps = Matrix(FP[indices,9:end])

        function leverage_dist(unique_comps_fps, Norman)
            stable = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * stable * x
                println(j)
            end
            return lev
        end
        
        function cityblock_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
                println(j)
            end
            return lev
        end

        AD_projection = leverage_dist(unique_comps_fps,unique_comps_fps)
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        AD = AD_projection
        inchi_train, inchi_test = train_test_split(classes, test_size=0.20, random_state=random_state,stratify = round.(AD,digits = 1))

        indices_train = findall(x->x in inchi_train, FP[:,:INCHIKEY])
        indices_test = findall(x->x in inchi_test, FP[:,:INCHIKEY])
        FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])

        X_train = Matrix(hcat(FP[indices_train,:pH_aq],FP[indices_train,9:end]))
        X_test = Matrix(hcat(FP[indices_test,:pH_aq],FP[indices_test,9:end]))
        y_train = FP[indices_train,:logIE]
        y_test = FP[indices_test,:logIE]
        

        return X_train, X_test, y_train, y_test
    end


    z = zeros(itr,7)
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        FP1 = Matrix(hcat(FP[!,:pH_aq],FP[!,9:end]))
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            learn_rate = rand(learn_r)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
            X_train, X_test, y_train, y_test = split_classes(ESI)

                # Old way  
            # X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=state);
                # New way
            ## Regression ##
            reg = ()
            if grow == :Lossguide
                reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=state, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
            elseif grow == :SymmetricTree
                reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=state, grow_policy=:SymmetricTree, min_data_in_leaf=leaf, verbose=false)
            else error("Provide a valid grow policy (:Lossguide or :SymmetricTree)")
            end
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = learn_rate
            z[j,5] = score(reg, X_train, y_train)
            z[j,6] = score(reg, X_test, y_test)
            z[j,7] = size(X_test,1)/(size(X_train,1)+size(X_test,1))
            println("End of $j / $itr iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], learn_rate = z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

function FP_optim(ESI; iterations::Int=50, split_size::Float64=0.2)
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Set ranges for hyperparameters
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(20:40:80), vcat(collect(100:100:900),collect(1000:200:1400)))
    random_seed_r = collect(1:1:3)
    learn_r = vcat(collect(0.02:0.02:0.15), collect(0.2:0.2:0.8))
    l2_leaf_reg_r = vcat(collect(3:2:6),collect(10:5:15))
    
    z = DataFrame(leaves=[],trees=[],FP_type=[],learn_rate=[],state=[],accuracy_train=[],accuracy_test=[],actual_test_size=[])
    for i = 1:15
        # Load data files
        data_whole = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_$(ESI_name)_$i.csv", DataFrame)

        variables_df = hcat(data_whole[!,:pH_aq],data_whole[!,9:end], makeunique=true)
        variables = Matrix(variables_df)
        for j = 1:iterations
            # Pick one random hyperparameter
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            random_seed = rand(random_seed_r)
            learn_rate = rand(learn_r)
            l2_leaf_reg = rand(l2_leaf_reg_r)
            #New splitting
            classes = unique(data_whole[:,:INCHIKEY])
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
            reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide,min_data_in_leaf=leaf, verbose=false)
            fit!(reg, X_train, y_train)
            z_temp = DataFrame(leaves=leaf,trees=tree,FP_type=i,learn_rate=learn_rate,state=random_seed,accuracy_train=round(score(reg, X_train, y_train), digits=4),accuracy_test=round(score(reg, X_test, y_test), digits=4),actual_test_size=round(length(test_set_indices)/size(data_whole,1), digits=4))
            z = append!(z,z_temp)
            println("(ESI $ESI_name $i/15) End of $j / $iterations iterations")
        end
    end
    return z
end

optimization_pos = FP_optim(+1)
optimization_neg = FP_optim(-1)

CSV.write("/media/saer/Elements SE/Data Alex/IE_prediction/results/models/FP_Cat_HyperparameterOptimization_NoRSM_pos.csv", optimization_pos)
CSV.write("/media/saer/Elements SE/Data Alex/IE_prediction/results/models/FP_Cat_HyperparameterOptimization_NoRSM_neg.csv", optimization_neg)



# Find best optimized models for all the different types of fingerprints
neg1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_NoRSM_neg.csv", DataFrame)
neg2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_RSM0.3_neg.csv", DataFrame)
neg3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_In-Stratified_neg.csv", DataFrame)

neg1[:,:RSM] .= NaN
neg2[:,:RSM] .= 0.3
neg = vcat(neg1,neg2)

pos1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_NoRSM_pos.csv", DataFrame)
pos2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_RSM0.3_pos.csv", DataFrame)
pos3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP models\\FP_Cat_HyperparameterOptimization_In-Stratified_pos.csv", DataFrame)

pos1[:,:RSM] .= NaN
pos2[:,:RSM] .= 0.3
pos = vcat(pos1,pos2)

function find_max_per_unique(data)
    max_results = DataFrame()
    for i in unique(data[:,:FP_type])
        filtered_temp = filter(:FP_type => ==(i), data)
        max_row_temp = DataFrame(sort(filtered_temp,[:accuracy_test], rev=true)[1,:])
        append!(max_results,max_row_temp)
    end
    return max_results
end


best_neg = find_max_per_unique(neg3)
display(best_neg)

best_pos = find_max_per_unique(pos3)
display(best_pos)
best_pos[:,:accuracy_test]
hcat(best_pos[:,:accuracy_test], best_neg[:,:accuracy_test])

