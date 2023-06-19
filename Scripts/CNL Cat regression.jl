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
using JLD
using LaTeXStrings
using LinearAlgebra
using Random
using StatsBase
@sk_import linear_model: LinearRegression
cat = pyimport("catboost")

# CNL
function Stratified_CNL_model_LM(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)

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
        ESI_symbol = "-"
        tree = 1400
        learn_rate = 0.6
        random_seed = 3
        leaf = 10
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    elseif ESI == 1
        ESI_name = "POS"
        ESI_symbol = "+"
        tree = 100
        learn_rate = 0.03
        random_seed = 2
        leaf = 4
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
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

        # Stratified splitting
        classes = unique(data_whole[:,:INCHIKEY])
        train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]

    ## Cat Regression ##
    reg.fit(X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1.5]

    ## Linear regression model ##
    if ESI == +1 
        y_hat_train_CAT = reg.predict(X_train)     # y_hat_train
        y_hat_test_CAT = reg.predict(X_test)       # y_hat_test

        LinModel = LinearRegression(n_jobs=-1)      # Linear model
        ScikitLearn.fit!(LinModel, reshape(y_hat_train_CAT,:,1), y_train)
        y_train_hat_LM = LinModel.predict(reshape(y_hat_train_CAT,:,1))
        y_test_hat_LM = LinModel.predict(reshape(y_hat_test_CAT,:,1))
        R2_train = LinModel.score(reshape(y_train_hat_LM,:,1),y_train)
        R2_test = LinModel.score(reshape(y_test_hat_LM,:,1),y_test)

        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = R2_train   # Train set accuracy
        z3 = R2_test      # Test set accuracy
        z4 = y_train_hat_LM     # y_hat_train
        z5 = y_test_hat_LM   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual
    elseif ESI == -1 
        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = score(reg, X_train, y_train)   # Train set accuracy
        z3 = score(reg, X_test, y_test)      # Test set accuracy
        z4 = reg.predict(X_train)     # y_hat_train
        z5 = reg.predict(X_test)   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual    
    end

    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title="ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel="Experimental log(IE)", ylabel = "Predicted log(IE)")
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

importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = Stratified_CNL_model_LM(-1, allowplots=true, allowsave=true,showph=true);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = Stratified_CNL_model_LM(+1, allowplots=true, allowsave=true,showph=true);

importance_neg
accuracy_te_neg
importance_pos

# 20-Apr-2023 Linear model based on residuals of Catboost
ESI = +1
function Stratified_CNL_model_LM_of_residuals(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)

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
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "NEG"
        tree = 1400
        learn_rate = 0.6
        random_seed = 3
        leaf = 10
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    elseif ESI == +1
        ESI_name = "POS"
        tree = 100
        learn_rate = 0.03
        random_seed = 2
        leaf = 4
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
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

    # Stratified splitting
    classes = unique(data_whole[:,:INCHIKEY])
    train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]

    ## Cat Regression ##
    reg.fit(X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1.5]

    ## Linear regression model ##
    if ESI == +1 
        function R2_Sklearn(y_hat_,y_)
            u = sum((y_ - y_hat_) .^2)
            v = sum((y_ .- mean(y_)) .^ 2)
            r2_result = 1 - (u/v)
            return r2_result
        end
    
        y_hat_train_CAT = reg.predict(X_train)     # y_hat_train
        y_hat_test_CAT = reg.predict(X_test)       # y_hat_test
        score_r2equation_train = R2_Sklearn(y_hat_train_CAT, y_train)
        score_r2equation_test = R2_Sklearn(y_hat_test_CAT, y_test)
        #score_catboost_train = reg.score(X_train, y_train)
        #score_catboost_test = reg.score(X_test, y_test)

        res_train = y_hat_train_CAT - y_train
        res_test = y_hat_test_CAT - y_test

        LinModel = LinearRegression(n_jobs=-1)      # Linear model
        ScikitLearn.fit!(LinModel, reshape(y_train,:,1), res_train)

        res_train_hat_LM = LinModel.predict(reshape(y_hat_train_CAT,:,1))
        res_test_hat_LM = LinModel.predict(reshape(y_hat_test_CAT,:,1))
        y_train_hat_LM = y_hat_train_CAT - res_train_hat_LM
        y_test_hat_LM = y_hat_test_CAT - res_test_hat_LM
        
        #R2_train = LinModel.score(reshape(y_train_hat_LM,:,1),y_train)
        #R2_test = LinModel.score(reshape(y_test_hat_LM,:,1),y_test)
        R2_train = R2_Sklearn(y_train_hat_LM, y_train)
        R2_test = R2_Sklearn(y_test_hat_LM, y_test)

        LM_plot_1 = scatter(y_hat_train_CAT, res_train, c=:green, xaxis ="Predicted logIE", yaxis="Residuals", label="Training set residual")
        scatter!(y_hat_test_CAT, res_test, c=:orange, label="Test set residual" )
        scatter!(y_train, res_train_hat_LM, c=:white, label="Training set predicted residual")
        scatter!(y_test, res_test_hat_LM, c=:yellow, label="Test set residual")
        display(LM_plot_1)
        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)


        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = R2_train   # Train set accuracy
        z3 = R2_test      # Test set accuracy
        z4 = y_train_hat_LM     # y_hat_train
        z5 = y_test_hat_LM   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual
    elseif ESI == -1 
        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = score(reg, X_train, y_train)   # Train set accuracy
        z3 = score(reg, X_test, y_test)      # Test set accuracy
        z3_ = reg.score(X_test, y_test)
        z4 = reg.predict(X_train)     # y_hat_train
        z5 = reg.predict(X_test)   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual    
    end

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

importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = Stratified_CNL_model_LM_of_residuals(-1, allowplots=true, allowsave=true,showph=true);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = Stratified_CNL_model_LM_of_residuals(+1, allowplots=true, allowsave=true,showph=true);

# 23-Apr-2023 Multiple models approach
ESI = +1
function Stratified_CNL_model_multiple(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)
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
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "NEG"
        tree = 1400
        learn_rate = 0.6
        random_seed = 3
        leaf = 10
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    elseif ESI == 1
        ESI_name = "POS"
        tree = 100
        learn_rate = 0.03
        random_seed = 2
        leaf = 4
        reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
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
    #
        # Stratified splitting
        classes = unique(data_whole[:,:INCHIKEY])
        train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)
        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])
    
        X_test = variables[test_set_indices,:]
        y_test = data_whole[test_set_indices,:logIE]
    
    
    ## Model A, B, C training set split
    X_train = hcat(data_whole[train_set_indices,[:INCHIKEY,:pH_aq]], data_whole[train_set_indices,8:end])
    y_train =  data_whole[train_set_indices,:logIE]

    classes_train = unique(X_train[:,:INCHIKEY])
    scatter(X_train[:,:INCHIKEY], y_train,size=(1200,800))

    avg_IEs_perINCHIKEY = sort(combine(groupby(DataFrame(INCHIKEY = X_train[:,:INCHIKEY], logIE = y_train), :INCHIKEY), :logIE => mean => :logIE_mean), [:logIE_mean])

    cutoff_A = 0.4
    modelA_inchikeys = avg_IEs_perINCHIKEY[1:Int(round(cutoff_A*(size(avg_IEs_perINCHIKEY,1)))),:INCHIKEY]
    modelC_inchikeys = avg_IEs_perINCHIKEY[end-Int(round(cutoff_A*(size(avg_IEs_perINCHIKEY,1)))):end,:INCHIKEY]
    modelB_inchikeys = avg_IEs_perINCHIKEY[Int(round((size(avg_IEs_perINCHIKEY,1)/2) - (cutoff_A*(size(avg_IEs_perINCHIKEY,1))/2))):Int(round((size(avg_IEs_perINCHIKEY,1)/2) + (cutoff_A*(size(avg_IEs_perINCHIKEY,1))/2))) ,:INCHIKEY]

    train_set_A_indices = findall(x -> x in modelA_inchikeys, data_whole[:,:INCHIKEY])
    train_set_B_indices = findall(x -> x in modelB_inchikeys, data_whole[:,:INCHIKEY])
    train_set_C_indices = findall(x -> x in modelC_inchikeys, data_whole[:,:INCHIKEY])

    X_train_A = hcat(data_whole[train_set_A_indices,[:INCHIKEY,:pH_aq]], data_whole[train_set_A_indices,8:end])
    y_train_A =  data_whole[train_set_A_indices,:logIE]
    X_train_B = hcat(data_whole[train_set_B_indices,[:INCHIKEY,:pH_aq]], data_whole[train_set_B_indices,8:end])
    y_train_B =  data_whole[train_set_B_indices,:logIE]
    X_train_C = hcat(data_whole[train_set_C_indices,[:INCHIKEY,:pH_aq]], data_whole[train_set_C_indices,8:end])
    y_train_C =  data_whole[train_set_C_indices,:logIE]

    function leverage_test(zz, Norman)
            x = Norman[j,:]
            lev = transpose(x) * zz * x
        return lev
    end
    function R2_eq(y_hat_,y_)
        u = sum((y_ - y_hat_) .^2)
        v = sum((y_ .- mean(y_)) .^ 2)
        r2_result = 1 - (u/v)
        return r2_result
    end

    reg_A = cat.CatBoostRegressor(verbose=false)
    reg_B = cat.CatBoostRegressor(verbose=false)
    reg_C = cat.CatBoostRegressor(verbose=false)

    reg_A.fit(Matrix(X_train_A[:,2:end]), y_train_A)
    reg_B.fit(Matrix(X_train_B[:,2:end]), y_train_B)
    reg_C.fit(Matrix(X_train_C[:,2:end]), y_train_C)

    y_train_hat_A = reg_A.predict(Matrix(X_train_A[:,2:end]))
    y_train_hat_B = reg_B.predict(Matrix(X_train_B[:,2:end]))
    y_train_hat_C = reg_C.predict(Matrix(X_train_C[:,2:end]))

    r2score_train_A = R2_eq(y_train_hat_A, y_train_A)
    r2score_train_B = R2_eq(y_train_hat_B, y_train_B)
    r2score_train_C = R2_eq(y_train_hat_C, y_train_C)

    zz_A = pinv(transpose(Int.(round.(Matrix(X_train_A[:,2:end])))) * (Int.(round.(Matrix(X_train_A[:,2:end])))))     # Maybe we can save and load those with a CSV file
    BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Leverage matrices\\zz_A", zz_A)
    zz_B = pinv(transpose(Int.(round.(Matrix(X_train_B[:,2:end])))) * (Int.(round.(Matrix(X_train_B[:,2:end])))))     # Maybe we can save and load those with a CSV file
    BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Leverage matrices\\zz_B", zz_B)
    zz_C = pinv(transpose(Int.(round.(Matrix(X_train_C[:,2:end])))) * (Int.(round.(Matrix(X_train_C[:,2:end])))))     # Maybe we can save and load those with a CSV file
    BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Leverage matrices\\zz_C", zz_C)

    # Approach: The test set are classified in the models independently (ie. the same compound can be in different models, depending on the CNLs)
    leverages = DataFrame(logIE = y_test, lev_test_A = ones(length(y_test)), lev_test_B = ones(length(y_test)), lev_test_C = ones(length(y_test)))
    categories = 4 .* ones(length(y_test))
    for i = 1:length(y_test)
        leverages[i,:lev_test_A] = transpose(X_test[i,:]) * zz_A * X_test[i,:]
        leverages[i,:lev_test_B] = transpose(X_test[i,:]) * zz_B * X_test[i,:]
        leverages[i,:lev_test_C] = transpose(X_test[i,:]) * zz_C * X_test[i,:]
        categories[i] = argmin([leverages[i,:lev_test_A], leverages[i,:lev_test_B], leverages[i,:lev_test_C]])
        println("$i/$(length(y_test))")
    end

    BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Leverage matrices\\categories", categories)
    X_test_set_A_indices = findall(x->x in "A", categories)
    X_test_set_B_indices = findall(x->x in "B", categories)
    X_test_set_C_indices = findall(x->x in "C", categories)

    y_test_hat_A = reg_A.predict(X_test[X_test_set_A_indices,:])
    y_test_hat_B = reg_B.predict(X_test[X_test_set_B_indices,:])
    y_test_hat_C = reg_C.predict(X_test[X_test_set_C_indices,:])
    y_test_A = y_test[X_test_set_A_indices]
    y_test_B = y_test[X_test_set_B_indices]
    y_test_C = y_test[X_test_set_C_indices]

    r2score_test_A = R2_eq(y_test_hat_A, y_test_A)
    r2score_test_B = R2_eq(y_test_hat_B, y_test_B)
    r2score_test_C = R2_eq(y_test_hat_C, y_test_C)



    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1.5]

    ## Linear regression model ##
    if ESI == +1 
        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = R2_train   # Train set accuracy
        z3 = R2_test      # Test set accuracy
        z4 = y_train_hat_LM     # y_hat_train
        z5 = y_test_hat_LM   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual
    elseif ESI == -1 
        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = score(reg, X_train, y_train)   # Train set accuracy
        z3 = score(reg, X_test, y_test)      # Test set accuracy
        z3_ = reg.score(X_test, y_test)
        z4 = reg.predict(X_train)     # y_hat_train
        z5 = reg.predict(X_test)   # y_hat_test
        z6 = z4 - y_train      # Train set residual
        z7 = z5 - y_test        # Test set residual    
    end

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


# 18-May-2023 Consensus spectra, Binning
function Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(ESI; consensus_threshold, allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, removeminusones::Bool=false, consensus_algorithm="replacement")
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
    function split_classes(ESI, classes; random_state::Int=random_seed, split_size::Float64=0.2)
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
        inchi_train = []   
        inchi_test = []   
        try
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))  
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)  
        end  
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "neg"
        ESI_symbol = "-"
        reg = cat.CatBoostRegressor(n_estimators=1600, learning_rate=0.15, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=10, verbose=false)
    elseif ESI == 1
        ESI_name = "pos"
        ESI_symbol = "+"
        reg = cat.CatBoostRegressor(n_estimators=1400, learning_rate=0.01, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=4, verbose=false)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    data_whole_raw = vcat(vcat(vcat(norman, amide), MB), NIST)

    # Scaling
    data_whole_raw[:,:MONOISOMASS] .= (data_whole_raw[:,:MONOISOMASS]) ./ 1000
    data_whole_raw[:,:pH_aq] .= (data_whole_raw[:,:pH_aq]) ./ 14

    # Removing the -1s
    if removeminusones == true
        for j = 1:size(data_whole_raw,1)
            data_whole_raw[j,[Vector(data_whole_raw[j,:]).==-1][1]] .= 0
            if (j % 50000) == 0
                println("Removing -1: $j/ $(size(data_whole_raw,1))")
            end
        end
    end

    # Filtering based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum = (count(==(1), Matrix(data_whole_raw[:,9:end]) ; dims=2))[:]
    data_whole_filtered = data_whole_raw[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum),:]
    
    # Split
    variables_df = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df)

    # Stratified splitting (Test set)
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])

    train = data_whole_filtered[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_test = data_whole_filtered[test_set_indices,:logIE]

    # Consensus spectra (replacing all spectra with one consensus spectrum)
    if (consensus_threshold > 0) && consensus_algorithm == "replacement"
        data_info = train[:,2:8]
        data_info_unique = unique(data_info)
        data_whole = DataFrame()
        for i = 1:size(data_info_unique,1)
            group_members = train[all(Matrix(train[:,2:8] .== DataFrame(data_info_unique[i,:])),dims=2)[:],:]
            min_freq = Int(floor(size(group_members,1)*consensus_threshold))

            if size(group_members,1) >= (3+min_freq)
                CNL_possible_indices = []
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    append!(CNL_possible_indices,ones_indices)                    
                end
                freq_dict = sort(countmap(CNL_possible_indices), rev=true)
                most_freq_CNLs = collect(keys(freq_dict))[collect(values(freq_dict) .>= min_freq)]

                consensus_spectrum_info = DataFrame(group_members[1,1:8])
                consensus_spectrum_cnl = DataFrame(group_members[1,9:end])
                consensus_spectrum_cnl[:,(Matrix(consensus_spectrum_cnl).==1)[:]] .= 0
                consensus_spectrum_i = hcat(consensus_spectrum_info, consensus_spectrum_cnl)
                consensus_spectrum_i[:,most_freq_CNLs] .= 1
                append!(data_whole, consensus_spectrum_i)
            else
                append!(data_whole, group_members)
            end
            if (i % 250) == 0
            println("$i/$(size(data_info_unique,1))")
            end
        end
    # Consensus spectra (removing fragments that are not present in the consensus spectrum)
    elseif (consensus_threshold > 0) && consensus_algorithm == "removal"
        data_info = train[:,2:8]
        data_info_unique = unique(data_info)
        data_whole = DataFrame()
        for i = 1:size(data_info_unique,1)
            group_members = train[all(Matrix(train[:,2:8] .== DataFrame(data_info_unique[i,:])),dims=2)[:],:]
            min_freq = Int(floor(size(group_members,1)*consensus_threshold))

            if size(group_members,1) >= (3+min_freq)
                CNL_possible_indices = []
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    append!(CNL_possible_indices,ones_indices)                    
                end
                freq_dict = sort(countmap(CNL_possible_indices), rev=true)
                most_freq_CNLs = Int.(collect(keys(freq_dict))[collect(values(freq_dict) .>= min_freq)] ) #These are the acceptable CNLs, everything else should be removed
                
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    ones_to_be_removed = setdiff(ones_indices, most_freq_CNLs)
                    group_members[j,ones_to_be_removed] .= 0
                end
                #consensus_spectrum_info = DataFrame(group_members[1,1:8])
                #consensus_spectrum_cnl = DataFrame(group_members[1,9:end])
                #consensus_spectrum_cnl[:,(Matrix(consensus_spectrum_cnl).==1)[:]] .= 0
                #consensus_spectrum_i = hcat(consensus_spectrum_info, consensus_spectrum_cnl)
                #consensus_spectrum_i[:,most_freq_CNLs] .= 1
                append!(data_whole, group_members)
            else
                append!(data_whole, group_members)
            end
            if (i % 250) == 0
            println("$i/$(size(data_info_unique,1))")
            end
        end
    else
        data_whole = train
    end

    # Filtering again based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum_afterConsensusFiltering = (count(==(1), Matrix(data_whole[:,9:end]) ; dims=2))[:]
    data_train = data_whole[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum_afterConsensusFiltering),:]

    # Splitting (Train set)
    X_train = Matrix(hcat(data_train[:,:pH_aq],data_train[:,8:end]))
    y_train =  data_train[:,:logIE]

    ## Regression ##
    ScikitLearn.fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]

    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual


    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2,dpi=300)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2,dpi=300)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2,dpi=300)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_Consensus_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
        scatter!(y_test,z7, label="Test set",color=:orange,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_Consensus_$ESI_name.png")
        end
        display(p123)
        display(p456)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2,dpi=300)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_Consensus_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet,dpi=300)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_Consensus_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end
importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(-1, min_CNLs=0, consensus_threshold=0.2, allowplots=true,allowsave=true, random_seed=3, removeminusones=true, showph=true) # ESI-
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(+1, min_CNLs=1, consensus_threshold=0.2, allowplots=true, allowsave=true,random_seed = 3,showph=true) # ESI+

importance_neg