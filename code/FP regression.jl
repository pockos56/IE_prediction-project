########################################################################
## Goal: Train the fingerprint (FP) model

using ScikitLearn, Plots, Statistics, DataFrames, CSV, PyCall, Conda, LaTeXStrings, LinearAlgebra, Random, ProgressBars
using ScikitLearn.CrossValidation: train_test_split
pcp = pyimport("pubchempy")
cat = pyimport("catboost")
jblb = pyimport("joblib")

## Paths
project_path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\"
path_data = joinpath(project_path, "data")
path_graphs = joinpath(project_path, "Graphs")

#Loading optimal hyperparameters for FP model
best_parameters_mean = sort(CSV.read(joinpath(path_data, "Optimised hyperparameters", "FP_optimization_mean_6(3).csv"), DataFrame),"accuracy_test", rev=true)

# Function to train the FP model and provide accuracy metrics
function FP_Cat_model_mode(mode::String; allowplots=false, allowsave=false, showph=false)
    # Load hyperparameters
    min_samples_per_leaf = best_parameters_mean[1,"leaves"]
    n_trees = best_parameters_mean[1,"trees"]
    learn_rate = best_parameters_mean[1,"learn_rate"]
    state = best_parameters_mean[1,"state"]
    depth = best_parameters_mean[1,"depth"]
    subsample = best_parameters_mean[1,"subsample"]
    colsample_bylevel = best_parameters_mean[1,"colsample_bylevel"]
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, depth=depth,colsample_bylevel=colsample_bylevel, subsample=subsample, verbose=false)

    FP = CSV.read(joinpath(path_data, "Fingerprints", "FP6_$mode.csv"), DataFrame)
    # Filter validation set compounds
    validation_inchikeys = CSV.read(joinpath(path_data, "Validation_set_inchikeys.csv"), DataFrame)
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
        CSV.write(joinpath(path_data, "FP_model_training set_$mode.csv"), training_set_to_save)
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
        p1 = scatter(y_train,z4,label="Training set", legend=:bottomright, title = "FP model", color = :lightblue1, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", markerstrokewidth=0.1, dpi=300)
        scatter!(y_test,z5,label="Test set", color=:orange, markerstrokewidth=0.1, dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=1.5,dpi=300)

        p2 = scatter(y_train,z4, legend=false, ticks=false, color = :lightblue1, alpha=0.8, markerstrokewidth=0.1, dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=1.5,dpi=300)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange, alpha=0.8, markerstrokewidth=0.1,dpi=300)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=1.5,dpi=300, c=:lightblue1)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        display(p123)
        if allowsave == true
            savefig(joinpath(path_graphs, "Fingerprints", "Cat_Regression_FP6_$mode.pdf"))
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "Regression residuals", color = :lightblue1, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
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
            savefig(joinpath(path_graphs, "Fingerprints", "Cat_Residuals_FP6_$mode.pdf"))
        end
        display(p456)
        if showph == true
            # Regression pH plot
            train_ind = findall(x->x .== "train", y_hat_df[:,"class_fp"])        
            test_ind = findall(x->x .== "test", y_hat_df[:,"class_fp"])        
            plot_pH = scatter(y_hat_df[train_ind,"IE"], y_hat_df[train_ind,"IE_hat_fp"],label="Training set", legend=:best, title = "FP model", markershape = :circle, marker_z = y_hat_df[train_ind,"pH_aq"], xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_hat_df[test_ind,"IE"], y_hat_df[test_ind,"IE_hat_fp"],label="Test set", marker_z = y_hat_df[test_ind,"pH_aq"], markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])], label="1:1 line",width=2,dpi=300)

            # Residual pH plot
            residuals_vec = y_hat_df[:,"IE_hat_fp"] - y_hat_df[:,"IE"]

            plot_pH_res = scatter(y_hat_df[train_ind,"IE"],residuals_vec[train_ind],label="Training set", legend=:best, title = "Residuals - FP model",markershape=:circle, marker_z=y_hat_df[train_ind,"pH_aq"],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_hat_df[test_ind,"IE"],residuals_vec[test_ind], markershape=:rect,marker_z=y_hat_df[test_ind,"pH_aq"], label="Test set",color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[3*std(residuals_vec),3*std(residuals_vec)],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[-3*std(residuals_vec),-3*std(residuals_vec)],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave == true
                savefig(joinpath(path_graphs, "Fingerprints", "Cat_Residuals_pH_FP6_$mode.pdf"))
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end

    if allowsave == true
        # Saving the models (joblib)
        jblb.dump(reg, joinpath(path_data, "models", "FP_reg_$mode.joblib"))
        # Saving the predicted IEs
        CSV.write(joinpath(path_data, "models", "y_hat_df_FP_$mode.csv"), y_hat_df)
    end
    return reg,importance,z1,z2,z3,z4,z5,z6,z7, y_hat_df
end

# Train model
reg, importance_percentage_mean, importance_mean, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, y_hat_df_mean = FP_Cat_model_mode("mean", allowplots=true, allowsave=false,showph=false);

# Residuals
meanRes_train = round((mean(abs.(res_train))), digits=2)
meanRes_test = round((mean(abs.(res_test))), digits=2)
RMSE_train = round(sqrt(mean(res_train.^2)), digits=2)
RMSE_test = round(sqrt(mean(res_test.^2)), digits=2)

# Variable mportance
importance_mean_df  = DataFrame(import_col_mean=importance_mean, importance_mean=round.(importance_percentage_mean[1:length(importance_mean)],digits=1))[1:10,:]

# Which compounds show the highest prediction errors?
y_hat_df_mean.residual = abs.(y_hat_df_mean.IE - y_hat_df_mean.IE_hat_fp)
y_hat_df_mean = sort(y_hat_df_mean, "residual",rev=true)

# Plot for pH distribution
using FreqTables
pH = (freqtable(round.(FP[:,"pH.aq."], digits=1)))
pH_freq = [values(pH)[i] for i in (1:length(pH))]
pH_values = names(pH)[1]
bar(pH_values, pH_freq, xlabel="pH", label="n=$(size(FP,1))", bars=sqrt(sum(pH_freq)), dpi=500)
savefig(joinpath(path_graphs, "pH distribution.pdf"))