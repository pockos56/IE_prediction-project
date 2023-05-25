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
using LinearAlgebra
using StatsBase
cat = pyimport("catboost")

# Stratified CNL model
function Stratified_CNL_optim(ESI; iterations::Int=20, split_size::Float64=0.2)
    function split_classes(ESI, classes; random_state::Int=1312, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_fps = Matrix(FP[indices,9:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * z * x
                #println(j)
            end
            return lev
        end
        
        function cityblock_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
                #println(j)
            end
            return lev
        end
    
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

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
        # Stratified splitting
        classes = unique(data_whole[:,:INCHIKEY])
        train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

        X_train = variables[train_set_indices,:]
        X_test = variables[test_set_indices,:]
        y_train =  data_whole[train_set_indices,:logIE]
        y_test = data_whole[test_set_indices,:logIE]
        println("$j / $iterations: Train/Test set split")

        ## Regression ##
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf,l2_leaf_reg = l2_leaf_reg, verbose=false)
        fit!(reg, X_train, y_train)

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

optimization_df_neg = Stratified_CNL_optim(-1, iterations=1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_NEG.CSV", optimization_df_neg)
optimization_df_pos = Stratified_CNL_optim(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_POS.CSV", optimization_df_pos)

# Strategy of multiple Models (an attempt to do sequential removal and leverage calculation)
function split_inchikeys_basedon_leverage(ESI; random_state::Int=1312, split_size::Float64=0.2)
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
        ZZ = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * ZZ * x
            #println(j)
        end
        return lev
    end
    
    #= This was an attempt to do sequential removal and leverage calculation
    first_threshold  = 0.3
    AD_first = DataFrame(INCHIKEY=FP[indices,:INCHIKEY],lev=leverage_dist(unique_comps_fps,unique_comps_fps))
    reliable_first_inchikeys = AD_first[(AD_first[:,:lev] .<= first_threshold),:INCHIKEY]
    reliable_first_indices = findall(x -> x in reliable_first_inchikeys, FP[indices,:INCHIKEY])
    unreliable_first_indices = findall(x -> x in (AD_first[(AD_first[:,:lev] .> first_threshold),:INCHIKEY]), FP[indices,:INCHIKEY])
    histogram(AD_first[:,:lev], bins=55)

    unique_comps_fps_second = Matrix(FP[unreliable_first_indices, 9:end])
    AD_second = DataFrame(INCHIKEY=FP[unreliable_first_indices,:INCHIKEY],lev=leverage_dist(unique_comps_fps_second,unique_comps_fps_second))
    histogram(AD_second[:,:lev], bins=55)
    display(AD_second)
    =#






    sort!(AD,[:lev])
    histogram(AD, bins=50)

    
    inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    return inchi_train, inchi_test
end

ESI = +1
# Just cutting the predicted IE in three Models
function Stratified_CNL_model(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)
    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println("Leverage: $j/$(size(Norman,1))")
        end
        return lev
    end

    function split_classes(ESI, classes; random_state::Int=1312, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_fps = Matrix(FP[indices,9:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * z * x
                #println(j)
            end
            return lev
        end
        
        function cityblock_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
                #println(j)
            end
            return lev
        end
    
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "NEG"
        tree = 1000
        learn_rate = 0.4
        random_seed = 3
        leaf = 8 
        reg = cat.CatBoostRegressor(n_estimators=1000, learning_rate=0.4, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=8, verbose=false)
    elseif ESI == 1
        ESI_name = "POS"
        tree = 600
        learn_rate = 0.05
        random_seed = 3
        leaf = 4
        l2_leaf_reg = 4
        reg = cat.CatBoostRegressor(n_estimators=600, learning_rate=0.05, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=4, l2_leaf_reg=4, rsm=0.3, verbose=false)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))
    NIST = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_Hadducts_selectedCNLs.CSV", DataFrame)), names(amide))

    data_whole = vcat(vcat(vcat(norman, amide), MB), NIST)
    variables_df = hcat(data_whole[!,:pH_aq],data_whole[!,8:end])
    variables = Matrix(variables_df)
    levs = leverage_dist(variables,variables)
    histogram(levs)

        # Stratified splitting
        classes = unique(data_whole[:,:INCHIKEY])
        train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]

    ## Regression ##
    fit!(reg, X_train, y_train)

    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg, X_train)     # y_hat_train
    z5 = predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual

    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_$ESI_name.png")
        end
        display(p123)
        display(p456)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end

ESI = +1
min_CNLs = 17
# Filtering the data based on a minimum number of CNLs per spectrum
function Stratified_CNL_model_wFiltering(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, tree::Int, learn_rate, leaf::Int)
    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println("Leverage: $j/$(size(Norman,1))")
        end
        return lev
    end
    function split_classes(ESI, classes; random_state::Int=1312, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_fps = Matrix(FP[indices,9:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * z * x
                #println(j)
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)       
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "NEG"
        ESI_symbol = "-"
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    elseif ESI == 1
        ESI_name = "POS"
        ESI_symbol = "+"
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))
    NIST = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_Hadducts_selectedCNLs.CSV", DataFrame)), names(amide))

    data_whole = vcat(vcat(vcat(norman, amide), MB), NIST)

    # Filtering based on minimum number of CNLs per spectrum
    prefiltered = Matrix(data_whole[:,9:end]) #Int.(Matrix(data_whole[:,9:end]))
    no_CNLs_per_spectrum = (count(==(1), prefiltered; dims=2))[:]
    #histogram(no_CNLs_per_spectrum, xlabel="No. of CNLs", ylabel="No. of spectra", legend=false)
    #histogram(no_CNLs_per_spectrum[no_CNLs_per_spectrum.<=10], xlabel="No. of CNLs", ylabel="No. of spectra", legend=false, xticks=(0:1:10))
    filtered_indices = findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum)

    data_whole_filtered = data_whole[filtered_indices,:]
    variables_df_filtered = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df_filtered)
    #levs = leverage_dist(variables,variables)
    #histogram(levs)

    # Stratified splitting
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole_filtered[train_set_indices,:logIE]
    y_test = data_whole_filtered[test_set_indices,:logIE]

    ## Regression ##
    fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]

    z1 = names(variables_df_filtered[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg, X_train)     # y_hat_train
    z5 = predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual

    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_$ESI_name.png")
        end
        display(p123)
        display(p456)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end

iterations = 10
leaf_r = collect(4:2:10)
tree_r = vcat([50,100], vcat(collect(200:200:800),(collect(1000:200:1600))))
random_seed_r = collect(1:1:3)
learn_r = [0.01,0.05,0.1,0.15,0.2]
range = vcat(collect(1:1:20), collect(25:5:30))

results_for_filtered_data = DataFrame(min_CNLs=Int.(zeros(iterations*length(range))),accuracy_train=zeros(iterations*length(range)),accuracy_test=zeros(iterations*length(range)),random_seed=Int.(zeros(iterations*length(range))),tree=Int.(zeros(iterations*length(range))),leaf=Int.(zeros(iterations*length(range))),learn_rate=zeros(iterations*length(range)))

for i = 1:length(range)
    for itr = 1:iterations
        leaf = rand(leaf_r)
        tree = rand(tree_r)
        random_seed = rand(random_seed_r)
        learn_rate = rand(learn_r)

        z1, accuracy_tr ,accuracy_te ,z4,z5,z6,z7 = Stratified_CNL_model_wFiltering(+1, min_CNLs=range[i], random_seed=random_seed, tree=tree, learn_rate=learn_rate, leaf=leaf)

        index_df = ((i-1)*iterations) + itr
        results_for_filtered_data[index_df,:min_CNLs] = range[i]
        results_for_filtered_data[index_df,:accuracy_train] = accuracy_tr
        results_for_filtered_data[index_df,:accuracy_test] = accuracy_te
        results_for_filtered_data[index_df,:random_seed] = random_seed
        results_for_filtered_data[index_df,:tree] = tree
        results_for_filtered_data[index_df,:leaf] = leaf
        results_for_filtered_data[index_df,:learn_rate] = learn_rate
        println("$index_df/$(iterations*(length(range))) Done!")
    end
end

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_accuracy_vs_minCNLs_optimizedin10iterations.CSV", results_for_filtered_data)


# Let's try binning the CNLs with a +- 20mDa
ESI = -1
split_size = 0.2
min_CNLs = 1
function Stratified_CNL_model_wFilteringBinning(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, tree::Int, learn_rate, leaf::Int)
    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println("Leverage: $j/$(size(Norman,1))")
        end
        return lev
    end
    function split_classes(ESI, classes; random_state::Int=1312, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_fps = Matrix(FP[indices,9:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * z * x
                #println(j)
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)       
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "NEG"
        ESI_symbol = "-"
        reg = cat.CatBoostRegressor(n_estimators=1000, learning_rate=0.1, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=6, verbose=false)
    elseif ESI == 1
        ESI_name = "POS"
        ESI_symbol = "+"
        reg = cat.CatBoostRegressor(n_estimators=1100, learning_rate=0.1, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=6, verbose=false)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))
    NIST = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_Hadducts_selectedCNLs.CSV", DataFrame)), names(amide))

    data_whole = vcat(vcat(vcat(norman, amide), MB), NIST)

    # Binning
    part_to_see = data_whole[1:100,:]
    names(part_to_see[:,9:end])





    # Filtering based on minimum number of CNLs per spectrum
    prefiltered = Matrix(data_whole[:,9:end]) #Int.(Matrix(data_whole[:,9:end]))
    no_CNLs_per_spectrum = (count(==(1), prefiltered; dims=2))[:]
    #histogram(no_CNLs_per_spectrum, xlabel="No. of CNLs", ylabel="No. of spectra", legend=false)
    #histogram(no_CNLs_per_spectrum[no_CNLs_per_spectrum.<=10], xlabel="No. of CNLs", ylabel="No. of spectra", legend=false, xticks=(0:1:10))
    filtered_indices = findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum)

    data_whole_filtered = data_whole[filtered_indices,:]
    variables_df_filtered = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df_filtered)
    #levs = leverage_dist(variables,variables)
    #histogram(levs)

    # Stratified splitting
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole_filtered[train_set_indices,:logIE]
    y_test = data_whole_filtered[test_set_indices,:logIE]

    ## Regression ##
    fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]

    z1 = names(variables_df_filtered[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg, X_train)     # y_hat_train
    z5 = predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual

    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_$ESI_name.png")
        end
        display(p123)
        display(p456)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end



# Function with df input, df output where the binning happens

function binning(df::DataFrame)
    return new_df
end
df = DataFrame("12.00" => [1, 0, 0, 1, 1],                "12.01" => [0, 1, 1, 0, 1],                "13.02" => [1, 0, 1, 0, 1],                "14.09" => [0, 1, 0, 1, 0],                "14.1" => [1, 0, 1, 0, 1])

# Define the threshold
threshold = 0.02

# Convert the column names to Float64, divide by the threshold, round to the nearest integer, and convert back to string
bin_names = [string(round(parse.(Float64,colname) / threshold)) for colname in names(df)]

# Create a new dataframe with the binned column names
df_new = DataFrame()

for (i, colname) in enumerate(bin_names)
    if !(colname in names(df_new))
        df_new[!, colname] = df[!, i]
    else
        df_new[!, colname] += df[!, i]
    end
end
function binning_colnames(df;threshold_mDa=20)
    threshold = threshold_mDa / 1000
    colnames_df = parse.(Float64, names(df))
    colnames_new = []
    for i = 1:length(colnames_df) 
        dist = abs.(colnames_df[i] .- colnames_df)
        colnames_tobemerged = string.(sort(colnames_df[dist .<= threshold]))
        if length(colnames_tobemerged) > 1
            colnames_new_temp = colnames_tobemerged[1]
            for i = 2:length(colnames_tobemerged)
                colnames_tobemerged
                colnames_new_temp = colnames_new_temp * "," * colnames_tobemerged[i]
            end
            push!(colnames_new, colnames_new_temp)
        elseif length(colnames_tobemerged) == 1
            push!(colnames_new, colnames_tobemerged[1])
        end
        println(i)
        #return colnames_new
    end  
    return unique(colnames_new)
end

df = data_whole[:,9:end] 
binning_colnames(df)
# Group columns together based on proximity
merged_cols = Dict()
for i in 1:length(colnames)
found = false
for j in keys(merged_cols)
if abs(colnames[i] - j) < threshold
push!(merged_cols[j], i)
found = true
break
end
end
if !found
merged_cols[colnames[i]] = [i]
end
end
               
               # Create new dataframe with merged columns
               new_df = DataFrame()
               for cols in values(merged_cols)
                   new_col = zeros(Int, size(df, 1))
                   for i in cols
                       new_col .= new_col .| df[:, i]
                   end
                   push!(new_df, new_col)
               end
               names!(new_df, ["Merged_$(i)" for i in 1:size(new_df, 2)])
               values(merged_cols)[2]


