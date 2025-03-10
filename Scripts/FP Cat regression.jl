## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
using JLD
using LaTeXStrings
using LinearAlgebra
using Random
#using CatBoost
cat = pyimport("catboost")
jblb = pyimport("joblib")

#
## Importance for FP-6 ##
function FP_Cat_model(ESI; allowplots=false, allowsave=false, showph=false)
    if ESI == -1
        ESI_name = "neg"
        n_trees = 1200
        learn_rate = 0.14
        state = 3
        min_samples_per_leaf = 4
    
    elseif ESI == 1
        ESI_name = "pos"
        n_trees = 1200
        learn_rate = 0.14
        state = 3
        min_samples_per_leaf = 6
    
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_6.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])
    

    # Old way  
   # X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=state);
    # New way
    function split_classes(ESI; random_state::Int=1312)
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
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                x = Norman[j,:]
                lev[j] = transpose(x) * z * x
                println(j)
            end
            return lev
        end
        
        function cityblock_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
                println(j)
            end
            return lev
        end

        AD_projection = leverage_dist(unique_comps_fps,unique_comps_fps)
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        AD = AD_projection
        inchi_train, inchi_test = train_test_split(classes, test_size=0.20, random_state=random_state,stratify = round.(AD,digits = 1))

        indices_train = findall(x->x in inchi_train, FP[:,:INCHIKEY])
        indices_test = findall(x->x in inchi_test, FP[:,:INCHIKEY])
        FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])

        X_train = hcat(FP[indices_train,:pH_aq],FP[indices_train,9:end])
        X_test = hcat(FP[indices_test,:pH_aq],FP[indices_test,9:end])
        y_train = FP[indices_train,:logIE]
        y_test = FP[indices_test,:logIE]
        

        return X_train, X_test, y_train, y_test
    end
    X_train, X_test, y_train, y_test = split_classes(ESI)

    # Modeling
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
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Fingerprints\\CatBoost\\Cat_Regression_M2M4_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Fingerprints\\CatBoost\\Cat_Residuals_M2M4_$ESI_name.png")
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
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Fingerprints\\CatBoost\\Cat_Regression_pHcolor_M2M4_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train[:,2],z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals",markershape=:circle, marker_z=y_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test[:,2],z7, markershape=:rect,marker_z=y_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\Fingerprints\\CatBoost\\Cat_Residuals_pHcolor_M2M4_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
return z1,z2,z3,z4,z5,z6,z7
end

importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = FP_Cat_model(-1, allowplots=true, allowsave=true,showph=true);
importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = FP_Cat_model(+1, allowplots=true, allowsave=true, showph=true);
importance_neg
importance_pos

# Upper and lower boundaries for different pHs
# Find the pH for maximum IE
BSON.@load "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\Cat_model_neg.bson" reg
BSON.@load "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\Cat_model_pos.bson" cat_pos

function pH_investigation(ESI; allowplots=false, allowsave=false)
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    # Create the model
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
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

pH_neg = pH_investigation(-1)
pH_pos = pH_investigation(+1)

names(pH_neg)

heatmap(pH_neg[:,:best_pH], pH_neg[:,:original_IE],pH_neg[:,:name])
sort(pH_neg[:,:best_pH])
IE_range = pH_neg[:,:highest_IE] - pH_neg[:,:lowest_IE]
df = DataFrame(pH = pH_neg[:,:pH_aq], Range = IE_range)

using StatsPlots
issame = pH_neg[:,:best_pH] .== pH_neg[:,:pH_aq]
histo1 = histogram(pH_neg[issame.==1,:original_IE], pH_neg[issame.==1,:pH_aq],c=1, bins=20, alpha=0.4)
histo2 = histogram!(pH_neg[issame.==0,:original_IE], pH_neg[issame.==0,:pH_aq],c=2, bins=20, alpha =0.4)
plot(histo1,histo2)

boxplot(Matrix(df)[:,1])


# Heatmap, pred. logIE, pH, 
# Correlation coefficient to see how different FPs affect pH

## Importance for FP-12 ##
function FP_Cat_model_2(ESI; allowplots=false, allowsave=false, showph=false)
    if ESI == -1
        ESI_name = "neg"
        ESI_symbol = "-"
        n_trees = 1200
        learn_rate = 0.14
        state = 1312 #3
        min_samples_per_leaf = 4
    elseif ESI == 1
        ESI_name = "pos"
        ESI_symbol = "+"
        n_trees = 1200
        learn_rate = 0.14
        state = 3
        min_samples_per_leaf = 6
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\Results\\padel_M2M4_$(ESI_name)_12.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,9:end])
    
    # New way
    function split_classes(FP; random_state::Int=1312, split_size::Float64=0.2)
        classes = unique(FP[:,:INCHIKEY])
        indices = Int.(zeros(length(classes)))
        for i = 1:length(classes)
            inchi_temp = classes[i]
            indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
        end
        unique_comps_FPs = Matrix(FP[indices,9:end])
    
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

    X_train = Matrix(FP1[train_set_indices,:])
    X_test = Matrix(FP1[test_set_indices,:])
    y_train = FP[train_set_indices,:logIE]
    y_test = FP[test_set_indices,:logIE]

    # Modeling
    reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf, verbose=false)
    fit!(reg, X_train, y_train)

    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    z1 = names(FP1[:,:])[importance_index[importance .>=1]]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg,X_train)     # y_hat_train
    z5 = predict(reg,X_test)   # y_hat_test
    #z6 = ((10 .^ z4) - (10 .^ y_train)) ./ (10 .^ z4)    # Train set residual
    #z7 = ((10 .^ z5) - (10 .^ y_test)) ./ (10 .^ z5)        # Test set residual
    z6 = z4 .- y_train    # Train set residual
    z7 = z5 .- y_test     # Test set residual
    
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from FP", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)", dpi=300)
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Fingerprints\\CatBoost\\Cat_Regression_M2M4_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Fingerprints\\CatBoost\\Cat_Residuals_M2M4_$ESI_name.png")
        end
        display(p456)
        if showph == true
            X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), Matrix(FP[!,[:pH_aq,:logIE]]), test_size=0.20, random_state=state);
            reg = cat.CatBoostRegressor(n_estimators=n_trees, learning_rate=learn_rate, random_state=state, grow_policy=:Lossguide, min_data_in_leaf=min_samples_per_leaf,verbose=false)
            fit!(reg, X_train, y_train[:,2])
            z4 = predict(reg,X_train)     # y_hat_train
            z5 = predict(reg,X_test)   # y_hat_test  
            z6 = z4 - y_train[:,2]      # Train set residual
            z7 = z5 - y_test[:,2]        # Test set residual
           
            
            plot_pH = scatter(y_train[:,2],z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from FP", markershape = :circle, marker_z = y_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet,dpi=300)
            scatter!(y_test[:,2],z5,label="Test set", marker_z = y_test[:,1] , markershape = :rect,color=:jet,dpi=300)
            plot!([minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))],[minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))], label="1:1 line",width=2,dpi=300)
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Fingerprints\\CatBoost\\Cat_Regression_pHcolor_M2M4_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train[:,2],z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=y_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual",dpi=300)
            scatter!(y_test[:,2],z7, markershape=:rect,marker_z=y_test[:,1], label="Test set",color=:jet,dpi=300)
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[0,0],label="1:1 line",width=2,dpi=300) # 1:1 line
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2,dpi=300) # +3 sigma
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2,dpi=300) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Fingerprints\\CatBoost\\Cat_Residuals_pHcolor_M2M4_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return reg,importance,z1,z2,z3,z4,z5,z6,z7
end

reg_neg, importance_percentage_neg, importance_neg, accuracy_tr_neg, accuracy_te_neg, y_hat_train_neg, y_hat_test_neg, res_train_neg, res_test_neg = FP_Cat_model_2(-1, allowplots=true, allowsave=false,showph=false);
reg_pos, importance_percentage_pos, importance_pos, accuracy_tr_pos, accuracy_te_pos, y_hat_train_pos, y_hat_test_pos, res_train_pos, res_test_pos = FP_Cat_model_2(+1, allowplots=true, allowsave=false,showph=false);

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
jblb.dump(reg_neg,"C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_reg_neg.joblib")
jblb.dump(reg_pos,"C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_reg_pos.joblib")

# Saving the models (BSON)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Models\\FP_reg_neg.bson", reg_neg)
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Models\\FP_reg_pos.bson", reg_pos)
