## import packages ##
using ScikitLearn, Plots, Statistics, DataFrames, CSV, PyCall, Conda, LaTeXStrings, LinearAlgebra, Random, ProgressBars
using ScikitLearn.CrossValidation: train_test_split
pcp = pyimport("pubchempy")
cat = pyimport("catboost")
jblb = pyimport("joblib")

#=Loading optimal hyperparameters for FP model
optim_min = sort(vcat(vcat(vcat(
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_1_16.csv", DataFrame),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_13.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_14_16.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_6(1).csv", DataFrame)),"accuracy_test", rev=true)
optim_mean = sort(vcat(vcat(vcat(
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_mean_1_16.csv", DataFrame),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_13.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_mean_14_16.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_mean_6(1).csv", DataFrame)),"accuracy_test", rev=true)
optim_max = sort(vcat(vcat(vcat(
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_max_1_12.csv", DataFrame),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_13.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_max_14_16.csv", DataFrame)),
    CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_max_6(1).csv", DataFrame)),"accuracy_test", rev=true)
highest_accuracies_min = groupby(optim_min, :FP_type) |> x -> combine(x, :accuracy_test => maximum => :Highest_TestSet_R2)
highest_accuracies_mean = groupby(optim_mean, :FP_type) |> x -> combine(x, :accuracy_test => maximum => :Highest_TestSet_R2)
highest_accuracies_max = groupby(optim_max, :FP_type) |> x -> combine(x, :accuracy_test => maximum => :Highest_TestSet_R2)
MMM = hcat(hcat(highest_accuracies_mean, highest_accuracies_min, makeunique=true),highest_accuracies_max, makeunique=true)
average_MMM = DataFrame("average_R2" => (mean(hcat(hcat(MMM[:,2],MMM[:,4]),MMM[:,6]),dims=2))[:])
best_fp = argmax(Matrix(average_MMM)[:])        # The fingerprint type with the highest average test set score

best_parameters_min_obsolete = hcat(DataFrame(optim_min[findfirst(x->x.== best_fp, optim_min[:,"FP_type"]),:]),DataFrame("l2_leaf_reg"=> 6))
best_parameters_mean_obsolete = hcat(DataFrame(optim_mean[findfirst(x->x.== best_fp, optim_mean[:,"FP_type"]),:]),DataFrame("l2_leaf_reg"=> 3))
best_parameters_max_obsolete = hcat(DataFrame(optim_max[findfirst(x->x.== best_fp, optim_max[:,"FP_type"]),:]),DataFrame("l2_leaf_reg"=> 3))
best_parameters_MMM = vcat(vcat(best_parameters_min,best_parameters_mean),best_parameters_max)
# With collimit and featurelimit
#best_parameters_min = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_6(2).csv", DataFrame),"accuracy_test", rev=true)
#best_parameters_mean = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_mean_6(2).csv", DataFrame),"accuracy_test", rev=true)
#best_parameters_max = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_max_6(2).csv", DataFrame),"accuracy_test", rev=true)
=#
best_parameters_min = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_min_6(3).csv", DataFrame),"accuracy_test", rev=true)
best_parameters_mean = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_mean_6(3).csv", DataFrame),"accuracy_test", rev=true)
best_parameters_max = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_max_6(3).csv", DataFrame),"accuracy_test", rev=true)

function FP_Cat_model_mode(mode::String; allowplots=false, allowsave=false, showph=false)
    if mode == "min"
        min_samples_per_leaf = best_parameters_min[1,"leaves"]
        n_trees = best_parameters_min[1,"trees"]
        learn_rate = best_parameters_min[1,"learn_rate"]
        state = best_parameters_min[1,"state"]
        depth = best_parameters_min[1,"depth"]
        subsample = best_parameters_min[1,"subsample"]
        colsample_bylevel = best_parameters_min[1,"colsample_bylevel"]
        reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, depth=depth,colsample_bylevel=colsample_bylevel, subsample=subsample, verbose=false)
    elseif mode == "mean"
        min_samples_per_leaf = best_parameters_mean[1,"leaves"]
        n_trees = best_parameters_mean[1,"trees"]
        learn_rate = best_parameters_mean[1,"learn_rate"]
        state = best_parameters_mean[1,"state"]
        depth = best_parameters_mean[1,"depth"]
        subsample = best_parameters_mean[1,"subsample"]
        colsample_bylevel = best_parameters_mean[1,"colsample_bylevel"]
        reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, depth=depth,colsample_bylevel=colsample_bylevel, subsample=subsample, verbose=false)
    elseif mode == "max"
        min_samples_per_leaf = best_parameters_max[1,"leaves"]
        n_trees = best_parameters_max[1,"trees"]
        learn_rate = best_parameters_max[1,"learn_rate"]
        state = best_parameters_max[1,"state"]
        depth = best_parameters_max[1,"depth"]
        subsample = best_parameters_max[1,"subsample"]
        colsample_bylevel = best_parameters_max[1,"colsample_bylevel"]
        reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, depth=depth,colsample_bylevel=colsample_bylevel, subsample=subsample, verbose=false)
    else error("Set mode to min, max, or mean")
    end

    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP6_$mode.csv", DataFrame)
    # Filter validation set compounds
    validation_inchikeys = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Validation_set_inchikeys.csv", DataFrame)
    deleteat!(FP, findall(x -> x in validation_inchikeys.INCHIKEY, FP.INCHIKEY))
    
    FP1 = hcat(FP[!,"pH.aq."],FP[!,10:end])
    
    function split_classes(FP; random_state::Int, split_size::Float64=0.2)
        classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            indices[i] = Int(findfirst(x->x .== classes[i], FP[:,:INCHIKEY]))
        end
        unique_comps_FPs = Matrix(FP[indices,10:end])
    
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
        inchi_train = []
        inchi_test = []
        
        try         
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)
        end
        return inchi_train, inchi_test
    end
    train_set_inchikeys, test_set_inchikeys = split_classes(FP, random_state=state)
    test_set_indices = findall(x -> x in test_set_inchikeys, FP[:,:INCHIKEY])
    train_set_indices = findall(x -> x in train_set_inchikeys, FP[:,:INCHIKEY])

    if allowsave
        training_set_to_save = FP[train_set_indices,:]
        unique(FP[train_set_indices,:INCHIKEY])
        CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_model_training set_$mode.csv", training_set_to_save)
    end
    X_train = Matrix(FP1[train_set_indices,:])
    X_test = Matrix(FP1[test_set_indices,:])
    y_train = FP[train_set_indices,:unified_IEs]
    y_test = FP[test_set_indices,:unified_IEs]
    y_hat_df = DataFrame("INCHIKEY" => FP[:,"INCHIKEY"], "pH_aq" => FP[:,"pH.aq."], "IE" => FP[:,"unified_IEs"], "IE_hat_fp" => -Inf*ones(length(FP[:,"INCHIKEY"])), "class_fp" => "tbd")

    # Modeling
    ScikitLearn.fit!(reg, X_train, y_train)

    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    z1 = names(FP1[:,:])[importance_index[importance .>=1]]   # Most important descriptors
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg,X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg,X_test)   # y_hat_test
    z6 = z4 .- y_train    # Train set residual
    z7 = z5 .- y_test     # Test set residual
    y_hat_df[train_set_indices, "IE_hat_fp"] = ScikitLearn.predict(reg,X_train)
    y_hat_df[test_set_indices, "IE_hat_fp"] = ScikitLearn.predict(reg,X_test)
    y_hat_df[train_set_indices, "class_fp"] .= "train"
    y_hat_df[test_set_indices, "class_fp"] .= "test"

    if allowplots
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "$mode IEs from FP", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", dpi=300)
        scatter!(y_test,z5,label="Test set", color=:orange,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2,dpi=300)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2,dpi=300)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2,dpi=300)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        display(p123)
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Regression_FP6_$mode.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
        scatter!(y_test,z7, label="Test set",color=:orange,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false,dpi=300)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Residuals_FP6_$mode.png")
        end
        display(p456)
        if showph == true
            # Regression pH plot
            train_ind = findall(x->x .== "train", y_hat_df[:,"class_fp"])        
            test_ind = findall(x->x .== "test", y_hat_df[:,"class_fp"])        
            plot_pH = scatter(y_hat_df[train_ind,"IE"], y_hat_df[train_ind,"IE_hat_fp"],label="Training set", legend=:best, title = "$mode IEs from FP", markershape = :circle, marker_z = y_hat_df[train_ind,"pH_aq"], xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_hat_df[test_ind,"IE"], y_hat_df[test_ind,"IE_hat_fp"],label="Test set", marker_z = y_hat_df[test_ind,"pH_aq"], markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])], label="1:1 line",width=2,dpi=300)
            if allowsave == true
                #savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Regression_pH_FP6_$mode.png")
            end
            # Residual pH plot
            residuals_vec = y_hat_df[:,"IE_hat_fp"] - y_hat_df[:,"IE"]

            plot_pH_res = scatter(y_hat_df[train_ind,"IE"],residuals_vec[train_ind],label="Training set", legend=:best, title = "Residuals - $mode FP model",markershape=:circle, marker_z=y_hat_df[train_ind,"pH_aq"],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_hat_df[test_ind,"IE"],residuals_vec[test_ind], markershape=:rect,marker_z=y_hat_df[test_ind,"pH_aq"], label="Test set",color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[3*std(residuals_vec),3*std(residuals_vec)],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[-3*std(residuals_vec),-3*std(residuals_vec)],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Residuals_pH_FP6_$mode.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end

    if allowsave == true
        # Saving the models (joblib)
        jblb.dump(reg, "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\FP_reg_$mode.joblib")
        # Saving the predicted IEs
        CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_$mode.csv", y_hat_df)
    end
    return reg,importance,z1,z2,z3,z4,z5,z6,z7, y_hat_df
end

reg, importance_percentage_min, importance_min, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, y_hat_df_min = FP_Cat_model_mode("min", allowplots=true, allowsave=true,showph=true);
reg, importance_percentage_mean, importance_mean, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, y_hat_df_mean = FP_Cat_model_mode("mean", allowplots=true, allowsave=true,showph=true);
reg, importance_percentage_max, importance_max, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, y_hat_df_max = FP_Cat_model_mode("max", allowplots=true, allowsave=true,showph=true);

# Residuals
meanRes_train = round((mean(abs.(res_train))), digits=2)
meanRes_test = round((mean(abs.(res_test))), digits=2)
RMSE_train = round(sqrt(mean(res_train.^2)), digits=2)
RMSE_test = round(sqrt(mean(res_test.^2)), digits=2)
#
# Importance tables
importance_min_df  = DataFrame(import_col_min=importance_min, importance_min=round.(importance_percentage_min[1:length(importance_min)],digits=1))[1:10,:]
importance_mean_df  = DataFrame(import_col_mean=importance_mean, importance_mean=round.(importance_percentage_mean[1:length(importance_mean)],digits=1))[1:10,:]
importance_max_df  = DataFrame(import_col_max=importance_max, importance_max=round.(importance_percentage_max[1:length(importance_max)],digits=1))[1:10,:]
#
# Which compounds show the highest prediction errors?
y_hat_df_mean.residual = abs.(y_hat_df_mean.IE - y_hat_df_mean.IE_hat_fp)
y_hat_df_mean = sort(y_hat_df_mean, "residual",rev=true)

# Is there a correlation between prediction errors and MW?
y_hat_df_mean.MW .= 0.0
for i in ProgressBar(1:size(y_hat_df_mean,1))
    try
        y_hat_df_mean.MW[i] = parse(Float64, pcp.get_compounds(y_hat_df_mean[i,"INCHIKEY"], "inchikey")[1].molecular_weight)
    catch
        continue
    end
end
y_hat_df_mean.MW = MW
scatter(y_hat_df_mean.residual, y_hat_df_mean.MW, ylabel="Residual (in log units)", xlabel="Molecular weight")
cor(y_hat_df_mean.MW, y_hat_df_mean.residual)
# i =1
# Is there a correlation between prediction errors and molecular complexity?
y_hat_df_mean.complexity .= 0.0
for i in ProgressBar(1:size(y_hat_df_mean,1))
    try
        y_hat_df_mean.complexity[i] = convert(Float64, pcp.get_compounds(y_hat_df_mean[i,"INCHIKEY"], "inchikey")[1].complexity)
    catch
        continue
    end
end
complexity_mean = mean(y_hat_df_mean.complexity)
complexity_median = median(y_hat_df_mean.complexity)
complexity_of_highest_error_comps_in_CNL_mean = [203, 501, 132, 92.9, 105, 303, 577, 403, 371, 97]

using StatsPlots
boxplot(y_hat_df_mean.complexity)
scatter!(complexity_of_highest_error_comps_in_CNL_mean)
violin!(complexity_of_highest_error_comps_in_CNL_mean)
#
# Is there a correlation between prediction errors and H donor count?
y_hat_df_mean.H_donor .= 0.0
y_hat_df_mean.H_donor[1] = 
(pcp.get_compounds(y_hat_df_mean[1,"INCHIKEY"], "inchikey")[1]).listkey
y_hat_df_mean.H_donor[1] = (pcp.get_compounds(y_hat_df_mean[1,"INCHIKEY"], "inchikey")[1]).Hbondacceptorcount
pcp.get_compounds(y_hat_df_mean[1,"INCHIKEY"], "inchikey")[1].record
for i in ProgressBar(1:size(y_hat_df_mean,1))
    try
        y_hat_df_mean.complexity[i] = convert(Float64, pcp.get_compounds(y_hat_df_mean[i,"INCHIKEY"], "inchikey")[1].complexity)
    catch
        continue
    end
end
complexity_mean = mean(y_hat_df_mean.complexity)
complexity_median = median(y_hat_df_mean.complexity)
complexity_of_highest_error_comps_in_CNL_mean = [203, 501, 132, 92.9, 105, 303, 577, 403, 371, 97]

using StatsPlots
boxplot(y_hat_df_mean.complexity)
scatter!(complexity_of_highest_error_comps_in_CNL_mean)
violin!(complexity_of_highest_error_comps_in_CNL_mean)
#

