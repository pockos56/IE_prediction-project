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
using Random
cat = pyimport("catboost")

# CNL
function CNL_model(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)
    if ESI == -1
        ESI_name = "NEG"
        n_trees = 800
        learn_rate = 0.14
        state = 3
        leaf = 10    
    elseif ESI == 1
        ESI_name = "POS"
        n_trees = 800
        learn_rate = 0.05
        state = 2
        leaf = 10
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)
    data_whole = vcat(vcat(norman, amide), MB)
    variables_df = hcat(data_whole[!,:pH_aq],data_whole[!,8:end])
    variables = Matrix(variables_df)
    MaxFeat = Int64(ceil(size(variables,2)/3))

    # Split
    #New splitting
    classes = unique(data_whole[:,3])
    test_set_inchikeys = classes[randperm(MersenneTwister(state),length(classes))[1:Int(round(split_size*length(classes)))]]
    train_set_inchikeys = classes[randperm(MersenneTwister(state),length(classes))[1+(Int(round(split_size*length(classes)))):end]]
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole[:,:INCHIKEY])
    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]

    #=Old Splitting
    classes = unique(data_whole[:,3])
    Random.seed!(random_state)
    test_set_indices = findall(x -> x in classes[rand(1:length(classes),Int(round(split_size * length(classes))))], data_whole[:,:INCHIKEY])
    train_set_indices = setdiff(collect(1:size(data_whole,1)),test_set_indices)
    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]
    
    (y_test[y_test.<-2.9])
    (y_train[y_train.<-2.9])
    sus_ind =findall(x->x.==-3.0,data_whole[:,:logIE])
    df = data_whole[sus_ind,:]
    =#

    ## Regression ##
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_seed=state, grow_policy=:Lossguide, min_data_in_leaf=leaf,verbose=false)
    fit!(reg, X_train, y_train)
    #if allowsave == true
    #    JLD.save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_model_$(ESI_name).jld", reg)
    #end
    importance = 100 .* sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]
    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg, X_train)     # y_hat_train
    z5 = predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual
    # Plots
    if allowplots == true
        plot1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_$ESI_name.png")
        end
        display(plot1)
        display(plot2)
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

importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = CNL_model(-1, allowplots=true, allowsave=true,showph=true);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = CNL_model(+1, allowplots=true, allowsave=true,showph=true);




# Mass & pH
function mass_pH_model(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, random_state::Int=666)
    if ESI == -1
        ESI_name = "NEG"
        n_trees = 800
        learn_rate = 0.07
        state = 2
        leaf = 6    
    elseif ESI == 1
        ESI_name = "POS"
        n_trees = 900
        learn_rate = 0.4
        state = 2
        leaf = 4
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs.CSV", DataFrame)[:,1:8]
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs.CSV", DataFrame)[:,1:8]
    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs.CSV", DataFrame)[:,1:8]
    data_whole = vcat(vcat(norman, amide), MB)
    variables_df = data_whole[!,[:pH_aq, :MONOISOMASS]]
    variables = Matrix(variables_df)

    # Split
    classes = unique(data_whole[:,3])
    Random.seed!(random_state)
    test_set_indices = findall(x -> x in classes[rand(1:length(classes),Int(round(split_size * length(classes))))], data_whole[:,:INCHIKEY])
    train_set_indices = setdiff(collect(1:size(data_whole,1)),test_set_indices)
    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  data_whole[train_set_indices,:logIE]
    y_test = data_whole[test_set_indices,:logIE]
    #X_train, X_test, y_train, y_test = train_test_split(variables, data_whole[!,:logIE], test_size=0.20, random_state=state);

    ## Regression ##
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_seed=state, grow_policy=:Lossguide, min_data_in_leaf=leaf,verbose=false)
    fit!(reg, X_train, y_train)
    #if allowsave == true
    #    JLD.save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_model_$(ESI_name).jld", reg)
    #end
    importance = 100 .* sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]
    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg, X_train)     # y_hat_train
    z5 = predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual
    # Plots
    if allowplots == true
        plot1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from Mass&pH", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Mass&pH\\Mass&pH_Cat_Regression_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals Mass&pH", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Mass&pH\\Mass&pH_Cat_Residuals_$ESI_name.png")
        end
        display(plot1)
        display(plot2)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from Mass&pH", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Mass&pH\\Mass&pH_Cat_Regression_pHcolor_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals Mass&pH",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Mass&pH\\Mass&pH_Cat_Residuals_pHcolor_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end
importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = mass_pH_model(-1, allowplots=true, allowsave=false,showph=false, random_state=1, split_size=0.25);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = mass_pH_model(+1, allowplots=true, allowsave=true,showph=true, random_state=1);
