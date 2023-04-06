## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using LinearAlgebra
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
cat = pyimport("catboost")
## load files ##

#=
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
=#

## Type-12 fingerprints optimization ##
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
ENV


