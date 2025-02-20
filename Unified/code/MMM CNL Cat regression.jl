## import packages ##
using ScikitLearn, Plots, Statistics, DataFrames, CSV, PyCall, Conda, LaTeXStrings, LinearAlgebra, Random, StatsBase
using ScikitLearn.CrossValidation: train_test_split
cat = pyimport("catboost")
jblb = pyimport("joblib")

#Loading optimal hyperparameters for CNL model
#optim_min_1 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_1.csv", DataFrame),"accuracy_test",rev=true)
#optim_min_2 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_2.csv", DataFrame),"accuracy_test",rev=true)
#optim_min_3 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_3.csv", DataFrame),"accuracy_test",rev=true)
#optim_min_4 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_4.csv", DataFrame),"accuracy_test",rev=true)
optim_min_5 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_5.csv", DataFrame),"accuracy_test",rev=true)
#optim_min_6 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_min_6.csv", DataFrame),"accuracy_test",rev=true)

#optim_mean_1 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_1.csv", DataFrame),"accuracy_test",rev=true)
#optim_mean_2 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_2.csv", DataFrame),"accuracy_test",rev=true)
#optim_mean_3 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_3.csv", DataFrame),"accuracy_test",rev=true)
#optim_mean_4 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_4.csv", DataFrame),"accuracy_test",rev=true)
optim_mean_5 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_5.csv", DataFrame),"accuracy_test",rev=true)
#optim_mean_6 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_mean_6.csv", DataFrame),"accuracy_test",rev=true)

#optim_max_1 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_1.csv", DataFrame),"accuracy_test",rev=true)
#optim_max_2 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_2.csv", DataFrame),"accuracy_test",rev=true)
#optim_max_3 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_3.csv", DataFrame),"accuracy_test",rev=true)
#optim_max_4 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_4.csv", DataFrame),"accuracy_test",rev=true)
optim_max_5 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_5.csv", DataFrame),"accuracy_test",rev=true)
#optim_max_6 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_max_6.csv", DataFrame),"accuracy_test",rev=true)

#optim_whole = reduce(vcat,[DataFrame(optim_min_3[1,:]), DataFrame(optim_mean_3[1,:]), DataFrame(optim_max_3[1,:])])
#optim_whole = reduce(vcat,[DataFrame(optim_min_4[1,:]), DataFrame(optim_mean_4[1,:]), DataFrame(optim_max_4[1,:])])
optim_whole = reduce(vcat,[DataFrame(optim_min_5[1,:]), DataFrame(optim_mean_5[1,:]), DataFrame(optim_max_5[1,:])])
#optim_whole = reduce(vcat,[DataFrame(optim_min_6[1,:]), DataFrame(optim_mean_6[1,:]), DataFrame(optim_max_6[1,:])])

## CNL model ##
function Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_mode(data_mode; allowplots=false, showph=false, allowsave=false)
    function split_classes(classes; random_state::Int=random_seed, split_size=0.2)
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Internal_pubchem_FPs_dict.csv", DataFrame)
        indices = [Int(findfirst(x->x .== classes[i], FP[:,:INCHIKEY])) for i in 1:length(classes)]
        unique_comps_fps = Matrix(FP[indices,2:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = transpose(Norman[j,:]) * z * Norman[j,:]
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)  
        inchi_train = []   #split_size=0.2
        inchi_test = []   #random_state=1
        try
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))  
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)  
        end  
        return inchi_train, inchi_test
    end
    # Load data files
    if data_mode == "min"
        optim = optim_whole[1,:]
    elseif data_mode == "mean"
        optim = optim_whole[2,:]
    elseif data_mode == "max"
        optim = optim_whole[3,:]
    else error("Please set mode to min, mean, or max")
    end

    data_whole_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero_$data_mode.csv",DataFrame)
    # Filter validation set compounds
    validation_inchikeys = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Validation_set_inchikeys.csv", DataFrame)
    deleteat!(data_whole_raw, findall(x -> x in validation_inchikeys.INCHIKEY, data_whole_raw.INCHIKEY))
    # Load optimized hyperparameters
    consensus_threshold = optim["consensus_thres"]
    consensus_algorithm = optim["consensus_alg"]
    min_CNLs = optim["min_CNLs"]
    random_seed = optim["random_seed"]
    reg = cat.CatBoostRegressor(n_estimators=optim["tree"], learning_rate=optim["learn_rate"], random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=optim["leaves"], depth=optim["depth"], subsample=optim["subsample"], colsample_bylevel=optim["colsample_bylevel"], verbose=false)

    # Scaling
    data_whole_raw[:,"MONOISOMASS"] .= (data_whole_raw[:,"MONOISOMASS"]) ./ 1000
    data_whole_raw[:,"pH_aq"] .= (data_whole_raw[:,"pH_aq"]) ./ 14
    # Filtering based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum = (count(==(1), Matrix(data_whole_raw[:,9:end]) ; dims=2))[:]
    data_whole_filtered = data_whole_raw[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum),:]
    # Split
    variables_df = hcat(data_whole_filtered[:,"pH_aq"],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df)

    # Stratified splitting (Test set)
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(classes; random_state=random_seed)

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,"INCHIKEY"])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,"INCHIKEY"])
    train = data_whole_filtered[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_test = data_whole_filtered[test_set_indices,"logIE"]

    # Consensus spectra (replacing all spectra with one consensus spectrum)
    if (consensus_threshold > 0) && consensus_algorithm == "replacement"
        data_info_unique = unique(train[:,2:8])
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
            if (i % 5000) == 0
            println("Replacing with consensus spectrum: $i/$(size(data_info_unique,1))")
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
                append!(data_whole, group_members)
            else
                append!(data_whole, group_members)
            end
            if (i % 5000) == 0
            println("Removing uncommon fragments: $i/$(size(data_info_unique,1))")
            end
        end
    else
        data_whole = train
    end

    # Filtering again based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum_afterConsensusFiltering = (count(==(1), Matrix(data_whole[:,9:end]) ; dims=2))[:]
    data_train = data_whole[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum_afterConsensusFiltering),:]

    # Splitting (Train set)
    X_train = Matrix(hcat(data_train[:,"pH_aq"],data_train[:,8:end]))
    y_train =  data_train[:,"logIE"]

    ## Regression ##
    ScikitLearn.fit!(reg, X_train, y_train)
   
    # Feature importance
    importance = sort(reg.feature_importances_, rev=true)
    significant_columns = sortperm(reg.feature_importances_, rev=true)[importance .>=1]
    
    #=
    feature_names = names(hcat(data_train[:,"pH_aq"],data_train[:,8:end]))
    importance = reg.feature_importances_
    importance_ = reg.get_feature_importance(type="Interaction", prettified=true)
    [feature_names[i] for i in (1 .+ parse.(Int,collect(importance_["Feature Id"])))]
    =#

    # Results
    z1 = names(variables_df[:,:])[significant_columns]   # Most important variables
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual
    y_hat_df_train = DataFrame("INCHIKEY"=>data_train[:,"INCHIKEY"], "pH_aq"=>14*data_train[:,"pH_aq"], "IE"=>data_train[:,"logIE"], "IE_hat_CNL"=>ScikitLearn.predict(reg, X_train), "class_CNL"=>"train")
    y_hat_df_test = DataFrame("INCHIKEY"=>data_whole_filtered[test_set_indices,"INCHIKEY"], "pH_aq"=>14*data_whole_filtered[test_set_indices,"pH_aq"], "IE"=>data_whole_filtered[test_set_indices,"logIE"], "IE_hat_CNL"=>ScikitLearn.predict(reg, X_test), "class_CNL"=>"test")
    y_hat_df = append!(y_hat_df_train,y_hat_df_test)

    highest_errors = deepcopy(y_hat_df)
    highest_errors.diff = highest_errors.IE_hat_CNL - highest_errors.IE
    highest_errors.diff_abs = abs.(highest_errors.IE_hat_CNL - highest_errors.IE)
    sort!(highest_errors, "diff_abs",rev=true)
    highest_errors = highest_errors[[findfirst(x->x in unique(highest_errors[:,:INCHIKEY])[i,:], highest_errors[:,:INCHIKEY]) for i in 1:10],:]

    # Plots
    if allowplots
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "$data_mode IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", dpi=300)
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
        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Regression_CNL_$(data_mode).png")
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

        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Residuals_$(data_mode).png")
        end
        display(p456)
        if showph
            train_ind = findall(x->x .== "train", y_hat_df[:,"class_CNL"])        
            test_ind = findall(x->x .== "test", y_hat_df[:,"class_CNL"])        
            plot_pH = scatter(y_hat_df[train_ind,"IE"], y_hat_df[train_ind,"IE_hat_CNL"],label="Training set", legend=:best, title = "$data_mode IEs from CNL", markershape = :circle, marker_z = y_hat_df[train_ind,"pH_aq"], xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_hat_df[test_ind,"IE"], y_hat_df[test_ind,"IE_hat_CNL"],label="Test set", marker_z = y_hat_df[test_ind,"pH_aq"], markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])], label="1:1 line",width=2,dpi=300)
            if allowsave
                #savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Regression_CNL_pH_$(data_mode).png")
            end
            # Residual Plots
            residuals_vec = y_hat_df[:,"IE_hat_CNL"] - y_hat_df[:,"IE"]


            plot_pH_res = scatter(y_hat_df[train_ind,"IE"],residuals_vec[train_ind],label="Training set", legend=:best, title = "Residuals - $(data_mode) CNL model",markershape=:circle, marker_z=y_hat_df[train_ind,"pH_aq"],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_hat_df[test_ind,"IE"],residuals_vec[test_ind], markershape=:rect,marker_z=y_hat_df[test_ind,"pH_aq"], label="Test set",color=:jet,dpi=300)
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[3*std(residuals_vec),3*std(residuals_vec)],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(y_hat_df[:,"IE"]),maximum(y_hat_df[:,"IE"])],[-3*std(residuals_vec),-3*std(residuals_vec)],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Residuals_CNL_pH_$(data_mode).png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    if allowsave
        # Saving the models (joblib)
        jblb.dump(reg, "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\CNL_reg_$data_mode.joblib")

        # Save the predicted IEs
        CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_$data_mode.csv", y_hat_df)
    end
    return reg,importance,z1,z2,z3,z4,z5,z6,z7, highest_errors
end

reg, importance_percentage_min, importance_min, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, high_error_comps_min = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_mode("min",allowplots=true, allowsave=true, showph=true);
reg, importance_percentage_mean, importance_mean, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, high_error_comps_mean = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_mode("mean",allowplots=true, allowsave=true, showph=true);
reg, importance_percentage_max, importance_max, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test, high_error_comps_max = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_mode("max",allowplots=true, allowsave=true, showph=true);

# Importance tables
importance_min_df  = DataFrame(import_col_min=importance_min, importance_min=round.(importance_percentage_min[1:length(importance_min)],digits=1))[1:10,:]
importance_mean_df  = DataFrame(import_col_mean=importance_mean, importance_mean=round.(importance_percentage_mean[1:length(importance_mean)],digits=1))[1:6,:]
importance_max_df  = DataFrame(import_col_max=importance_max, importance_max=round.(importance_percentage_max[1:length(importance_max)],digits=1))[1:6,:]
#
# Residuals
meanRes_train = round((mean(abs.(res_train))), digits=2)
meanRes_test = round((mean(abs.(res_test))), digits=2)
RMSE_train = round(sqrt(mean(res_train.^2)), digits=2)
RMSE_test = round(sqrt(mean(res_test.^2)), digits=2)
#
