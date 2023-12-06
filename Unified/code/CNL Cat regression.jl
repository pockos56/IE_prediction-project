## import packages ##
using ScikitLearn, Plots, Statistics, DataFrames, CSV, PyCall, Conda, LaTeXStrings, LinearAlgebra, Random
using ScikitLearn.CrossValidation: train_test_split
using CatBoost
cat = pyimport("catboost")
jblb = pyimport("joblib")

#Loading optimal hyperparameters for CNL model
opt_1 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_expanded_1.csv", DataFrame),"accuracy_test",rev=true)
opt_2 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_expanded_2.csv", DataFrame),"accuracy_test",rev=true)
opt_3 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_expanded_3.csv", DataFrame),"accuracy_test",rev=true)
opt_4 = sort(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL_optimization_expanded_4.csv", DataFrame),"accuracy_test",rev=true)

## CNL model ##
function Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_expanded(; allowplots=false, allowsave=false, showph=false)
    function split_classes(classes; random_state::Int=random_seed)
        predicted_inchikeys = unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL-IE_Expansion.csv", DataFrame)[:,"INCHIKEY"])
        experimental_inchikeys= unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero.csv", DataFrame)[:,"INCHIKEY"])
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Internal_pubchem_FPs_dict.csv", DataFrame)
        unique_fps_exp_df = FP[[Int(findfirst(x->x .== experimental_inchikeys[i], FP[:,:INCHIKEY])) for i in 1:length(experimental_inchikeys)],:]
        unique_fps_pre_df = FP[[Int(findfirst(x->x .== predicted_inchikeys[i], FP[:,:INCHIKEY])) for i in 1:length(predicted_inchikeys)],:]
        
        #Filter the ones that are not found in classes variables
        unique_fps_exp_filt = unique_fps_exp_df[findall(x->x in classes, unique_fps_exp_df[:,"INCHIKEY"]),:]
        unique_fps_pre_filt = unique_fps_pre_df[findall(x->x in classes, unique_fps_pre_df[:,"INCHIKEY"]),:]
        
        unique_fps_exp = Matrix(unique_fps_exp_filt[:,2:end])
        unique_fps_pre = Matrix(unique_fps_pre_filt[:,2:end])
        unique_inchikeys_exp = unique_fps_exp_filt[:,1]
        unique_inchikeys_pre = unique_fps_pre_filt[:,1]
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = transpose(Norman[j,:]) * z * Norman[j,:]
            end
            return lev
        end
            
        AD_exp = leverage_dist(unique_fps_exp,unique_fps_exp)  
        AD_pre = leverage_dist(unique_fps_pre,unique_fps_pre)  
        inchi_train_exp = []   
        inchi_test_exp = []
        try
            inchi_train_exp, inchi_test_exp = train_test_split(unique_inchikeys_exp, test_size=split_size, random_state=random_state,stratify = round.(AD_exp,digits = 1))  
        catch
            inchi_train_exp, inchi_test_exp = train_test_split(unique_inchikeys_exp, test_size=split_size, random_state=random_state)  
        end  
    
        inchi_train_pre = []   
        inchi_test_pre = []
        try
            inchi_train_pre, inchi_test_pre = train_test_split(unique_inchikeys_pre, test_size=split_size, random_state=random_state,stratify = round.(AD_pre,digits = 1))  
        catch
            inchi_train_pre, inchi_test_pre = train_test_split(unique_inchikeys_pre, test_size=split_size, random_state=random_state)  
        end  

        inchi_train = unique(vcat(inchi_train_exp, inchi_train_pre))
        inchi_test = unique(vcat(inchi_test_exp, inchi_test_pre))
        return inchi_train, inchi_test
    end
    split_size = 0.2
    consensus_threshold = 0
    consensus_algorithm = "replacement"
    min_CNLs = 0
    random_seed = 3
    reg = cat.CatBoostRegressor(n_estimators=600, learning_rate=0.01, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=4, verbose=false)

    # Load data files
    data_whole_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_after_expansion.csv",DataFrame)
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
    #reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    #delete
    reg = cat.CatBoostRegressor(n_estimators=600, learning_rate=0.01, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=4, verbose=false,per_float_feature_quantization=["1:border_count=1024","2:border_count=1024"])
    #
    ScikitLearn.fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    significant_columns = sortperm(reg.feature_importances_, rev=true)[importance .>=1]

    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual

    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", dpi=300)
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Regression_1.png")
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Residuals_1_exp.png")
        end
        display(p456)
        p7 = scatter(z4,z6,label="Training set", legend=:best, title = "Regression residuals", color = :magenta, xlabel = "Predicted log(IE)", ylabel = "Residual",dpi=300)
        scatter!(z5,z7, label="Test set",color=:orange,dpi=300)
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[0,0],label="pred = exp",width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
        p8 = scatter(z4,z6, legend=false, ticks=false, color = :magenta,dpi=300)
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma
        p9 = scatter(z5,z7, label="Test set",color=:orange,legend=false, ticks=false,dpi=300)
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[0,0],width=2,dpi=300) # 1:1 line
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) # +3 sigma
        plot!([minimum(vcat(z5,z4)),maximum(vcat(z5,z4))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2,dpi=300) #-3 sigma

        p789 = plot(p7,p8,p9,layout= @layout [a{0.7w} [b; c]])

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL\\Cat_Residuals_1_pred.png")
        end
        display(p789)

    end

    return reg,importance,z1,z2,z3,z4,z5,z6,z7
end

reg, importance_percentage, importance, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered_expanded(allowplots=true, allowsave=true,showph=false);

# Residuals
meanRes_train_neg = round(10^(mean(abs.(sort(res_train_neg)))), digits=3)
meanRes_test_neg = round(10^(mean(abs.(sort(res_test_neg)))), digits=3)
meanRes_train_pos = round(10^(mean(abs.(sort(res_train_pos)))), digits=3)
meanRes_test_pos =round(10^(mean(abs.(sort(res_test_pos)))), digits=3)
#
meanRes_train_neg = round((mean(abs.(res_train_neg))), digits=3)
meanRes_test_neg = round((mean(abs.(res_test_neg))), digits=3)
meanRes_train_pos = round((mean(abs.(res_train_pos))), digits=3)
meanRes_test_pos = round((mean(abs.(res_test_pos))), digits=3)
#
RMSE_train_neg = round(sqrt(mean(res_train_neg.^2)), digits=3)
RMSE_test_neg = round(sqrt(mean(res_test_neg.^2)), digits=3)
RMSE_train_pos = round(sqrt(mean(res_train_pos.^2)), digits=3)
RMSE_test_pos = round(sqrt(mean(res_test_pos.^2)), digits=3)

# Saving the models (joblib)
jblb.dump(reg, "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\FP_reg.joblib")

# Saving the models (BSON)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\FP_reg.bson", reg)
