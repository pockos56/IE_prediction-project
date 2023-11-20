## import packages ##
using ScikitLearn, BSON, Plots, Statistics, DataFrames, CSV, PyCall, Conda,LaTeXStrings, LinearAlgebra, Random
using ScikitLearn.CrossValidation: train_test_split
using CatBoost
cat = pyimport("catboost")
jblb = pyimport("joblib")

#Loading optimal hyperparameters for FP model
opt_whole = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_optimization_1_16.csv", DataFrame)
highest_accuracies = groupby(opt_whole, :FP_type) |> x -> combine(x, :accuracy_test => maximum => :Highest_TestSet_R2)
best_parameters = opt_whole[argmax(opt_whole[:,:accuracy_test]),:]

## Importance for FP-12 ##
function FP_Cat_model_2(;ESI::Int = +1, allowplots=false, allowsave=false, showph=false)
    if ESI == -1
    elseif ESI == 1
        n_trees = 1400
        learn_rate = 0.06
        state = 3
        min_samples_per_leaf = 4
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP6.csv", DataFrame)
    FP1 = hcat(FP[!,"pH.aq."],FP[!,10:end])
    
    # New way
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
    # DELETE
    #training_set_to_save = FP[train_set_indices,:]
    #unique(FP[train_set_indices,:INCHIKEY])
    #CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_model_training set.csv", training_set_to_save)
    #
    X_train = Matrix(FP1[train_set_indices,:])
    X_test = Matrix(FP1[test_set_indices,:])
    y_train = FP[train_set_indices,:unified_IEs]
    y_test = FP[test_set_indices,:unified_IEs]

    # Modeling
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, verbose=false)
    ScikitLearn.fit!(reg, X_train, y_train)

    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    z1 = names(FP1[:,:])[importance_index[importance .>=1]]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg,X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg,X_test)   # y_hat_test
    #z6 = ((10 .^ z4) - (10 .^ y_train)) ./ (10 .^ z4)    # Train set residual
    #z7 = ((10 .^ z5) - (10 .^ y_test)) ./ (10 .^ z5)        # Test set residual
    z6 = z4 .- y_train    # Train set residual
    z7 = z5 .- y_test     # Test set residual
    
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "IEs from FP", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", dpi=300)
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Regression_FP12.png")
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Residuals_FP12.png")
        end
        display(p456)
        if showph == true
            X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), Matrix(FP[!,["pH.aq.","unified_IEs"]]), test_size=0.20, random_state=state);
            reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf,verbose=false)
            ScikitLearn.fit!(reg, X_train, y_train[:,2])
            z4 = ScikitLearn.predict(reg,X_train)     # y_hat_train
            z5 = ScikitLearn.predict(reg,X_test)   # y_hat_test  
            z6 = z4 - y_train[:,2]      # Train set residual
            z7 = z5 - y_test[:,2]        # Test set residual
           
            
            plot_pH = scatter(y_train[:,2],z4,label="Training set", legend=:best, title = "IEs from FP", markershape = :circle, marker_z = y_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_test[:,2],z5,label="Test set", marker_z = y_test[:,1] , markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))],[minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))], label="1:1 line",width=2,dpi=300)
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Regression_pH_FP12.png")
            end

            plot_pH_res = scatter(y_train[:,2],z6,label="Training set", legend=:best, title = "Regression residuals",markershape=:circle, marker_z=y_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_test[:,2],z7, markershape=:rect,marker_z=y_test[:,1], label="Test set",color=:jet,dpi=300)
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\Fingerprints\\Cat_Residuals_pH_FP12.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return reg,importance,z1,z2,z3,z4,z5,z6,z7
end
#reg_neg, importance_percentage_neg, importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = FP_Cat_model_2(-1, allowplots=true, allowsave=false,showph=false);
#reg_pos, importance_percentage_pos, importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = FP_Cat_model_2(+1, allowplots=true, allowsave=false,showph=false);
reg, importance_percentage, importance, accuracy_tr, accuracy_te, y_hat_train, y_hat_test, res_train, res_test = FP_Cat_model_2(allowplots=true, allowsave=true,showph=true);

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
