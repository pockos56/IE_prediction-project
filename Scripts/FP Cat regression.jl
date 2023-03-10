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
#using JLD
cat = pyimport("catboost")

## Importance for Padel-12 ##
function parameter(ESI; allowplots=false, allowsave=false, showph=false)
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])
    n_trees = 600
    learn_rate = 0.1
    state = 3
    min_samples_per_leaf = 4
    
    X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=state);
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf,verbose=false)
    fit!(reg, X_train, y_train)
        if allowsave == true
           # JLD.@save "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\Cat_model_$(ESI_name).jld" reg
        end
    importance = 100 .* sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]
    z1 = names(FP1[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg,X_train)     # y_hat_train
    z5 = predict(reg,X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual
    
    if allowplots == true
        plot1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from FP", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_Regression_M2M4_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_Residuals_M2M4_$ESI_name.png")
        end
        display(plot1)
        display(plot2)
        if showph == true
            X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), Matrix(FP[!,[:pH_aq,:logIE]]), test_size=0.20, random_state=state);
            reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf,verbose=false)
            fit!(reg, X_train, y_train[:,2])
            z4 = predict(reg,X_train)     # y_hat_train
            z5 = predict(reg,X_test)   # y_hat_test  
            z6 = z4 - y_train[:,2]      # Train set residual
            z7 = z5 - y_test[:,2]        # Test set residual
           
            
            plot_pH = scatter(y_train[:,2],z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from FP", markershape = :circle, marker_z = y_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test[:,2],z5,label="Test set", marker_z = y_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))],[minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))], label="1:1 line",width=2)
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_Regression_pHcolor_M2M4_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train[:,2],z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals",markershape=:circle, marker_z=y_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test[:,2],z7, markershape=:rect,marker_z=y_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_Residuals_pHcolor_M2M4_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
return z1,z2,z3,z4,z5,z6,z7
end

importance_minus, accuracy_tr_minus, accuracy_te_minus, y_hat_train_minus, y_hat_test_minus, res_train_minus, res_test_minus = parameter(-1, allowplots=false, allowsave=true,showph=false);
importance_plus, accuracy_tr_plus, accuracy_te_plus, y_hat_train_plus, y_hat_test_plus, res_train_plus, res_test_plus = parameter(+1, allowplots=false, allowsave=true, showph=false);
importance_minus
importance_plus


## pH distribution
histogram(data_plus[:,:pH_aq], bins=20 ,label = "ESI -",xlims=(0,14))
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_pH distribution ESI+.png")
histogram(data_minus[:,:pH_aq], bins=20 ,label = "ESI +",xlims=(0,14))
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Cat_pH distribution ESI-.png")

# Upper and lower boundaries for different pHs
# Find the pH for maximum IE
BSON.@load "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\Cat_model_minus.bson" reg
BSON.@load "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\Cat_model_plus.bson" cat_plus


function pH_investigation(ESI; allowplots=false, allowsave=false)
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    # Create the model
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])
    n_trees = 600
    learn_rate = 0.1
    state = 3
    min_samples_per_leaf = 4
    
    X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=state);
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf,verbose=false)
    fit!(reg, X_train, y_train)

    #Predict for several pHs
    pH_targets = [2,4,6,7,8,10,12]
    results = DataFrame()

    for i = 1:size(FP,1)
        # pH targets (original pH and pH targets)
        pH_targets_temp = vcat(FP[i,:pH_aq],pH_targets)

        # Dataframe with duplicate rows of the comp to be predicted
        temp_original = DataFrame()
        for j = 1:length(pH_targets_temp)
            push!(temp_original, FP[i,:])
        end        
        temp_original.pH_aq = pH_targets_temp

        # pH_aq and fingerprints matrix
        FP_matrix = hcat(temp_original[!,:pH_aq], temp_original[!,9:end])

        # IE prediction
        IE_temp = zeros(length(pH_targets_temp))
        for k = 1:length(IE_temp)
            IE_temp[k] = reg.predict(Vector(FP_matrix[k,:]))
        end
        
        # Extracted variables
        lowest_IE = minimum(IE_temp)
        highest_IE = maximum(IE_temp)
        original_IE = IE_temp[1]
        best_pH = pH_targets_temp[sortperm(IE_temp,rev =true)[1]]

        # Reporting
        info_IE_temp = DataFrame(lowest_IE = minimum(IE_temp), highest_IE = maximum(IE_temp), original_IE = IE_temp[1], best_pH = pH_targets_temp[sortperm(IE_temp,rev =true)[1]])
        info_comp_temp = DataFrame(temp_original[1,1:8])
        results_temp = hcat(info_comp_temp,info_IE_temp)
        append!(results, results_temp)

        println("Compound $i out of $(size(FP,1))")
    end
    return results
end



pH_investigation(-1)



























## Cross Validation
CV = cross_val_score(reg, X_train, y_train, cv=5)
CV_mean = mean(CV)