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

## Find optimal parameters
hyper_neg = vcat(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_NEG.1.csv", DataFrame), CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_NEG.2.csv", DataFrame))
hyper_pos = vcat(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_POS.1.csv", DataFrame), CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_Cat_HyperparameterOptimization_POS.2.csv", DataFrame))



# CNL
function CNL_model(ESI; allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2)
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
    NIST = DataFrame(Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs.CSV", DataFrame)), names(amide))

    data_whole = vcat(vcat(vcat(norman, amide), MB), NIST)
    variables_df = hcat(data_whole[!,:pH_aq],data_whole[!,8:end])
    variables = Matrix(variables_df)

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

    ## Regression ##
    fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=3]
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

importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = CNL_model(-1, allowplots=true, allowsave=true,showph=true);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = CNL_model(+1, allowplots=true, allowsave=true,showph=true);
