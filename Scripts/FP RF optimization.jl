## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using LaTeXStrings
using LinearAlgebra
using Random
using DataFrames
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda

@sk_import ensemble: RandomForestRegressor
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

## load files ##
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI-.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI+.csv", DataFrame)
data3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI-.csv", DataFrame)
data4 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI+.csv", DataFrame)

data_M4_minus = (unique(data1,2))[!,[2,3,27,28,29,30,31]]
data_M4_plus = (unique(data2,2))[!,[2,3,27,28,29,30,31]]
data_M2_minus = data3[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_M2_plus = data4[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_minus = vcat(data_M4_minus, data_M2_minus)
data_plus = vcat(data_M4_plus, data_M2_plus)

## Custom FP parameter optimization ##
    coarseness_r = vcat(collect(100:100:1000), collect(1200:300:3000))
    leaf_r = collect(4:2:12)
    tree_r = vcat(collect(50:50:300),collect(400:100:1000))
    itr = 50
    z = zeros(itr*length(coarseness_r),6)
    for i = 1:length(coarseness_r)
        coarseness = coarseness_r[i]
        FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\FP_minus_$coarseness.csv", DataFrame))
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            MaxFeat = Int64(ceil(coarseness/3))

        ## Regression ##
            X_train, X_test, y_train, y_test = train_test_split(FP1, data_minus[1:end,3], test_size=0.20, random_state=state);
            reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
            fit!(reg, X_train, y_train)
            z[((i-1)*itr+j),1] = leaf
            z[((i-1)*itr+j),2] = tree
            z[((i-1)*itr+j),3] = coarseness
            z[((i-1)*itr+j),4] = state
            z[((i-1)*itr+j),5] = score(reg, X_train, y_train)
            z[((i-1)*itr+j),6] = score(reg, X_test, y_test)
    end
        println("End of $coarseness coarseness")
    end    

    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], coarseness = z[:,3], state=z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
    z_df_sorted_minus = sort(z_df, :accuracy_test, rev=true)
    CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\z_df_sorted_minus", z_df_sorted_minus)

    scatter(z_df_sorted_minus[:,:coarseness],z_df_sorted_minus[:,:accuracy_test], ylims =(0.5,0.6))


## Padel fingerprints optimization ##
    function optim(output, ESI, iterations=50)
        leaf_r = collect(4:2:10)
        tree_r = vcat(collect(50:50:300),collect(400:100:1000))
        itr = iterations
        if ESI == -1
            ESI_name = "minus"
        elseif ESI == 1
            ESI_name = "plus"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end

        z = zeros(itr*13,6)
        for i = 0:12
            FP1 = Matrix(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_$(ESI_name)_$i.csv", DataFrame))[:,2:end]
            for j = 1:itr
                leaf = rand(leaf_r)
                tree = rand(tree_r)
                state = rand(1:3)
                MaxFeat = Int64(ceil(size(FP1,2)/3))
            ## Regression ##
                X_train, X_test, y_train, y_test = train_test_split(FP1, output, test_size=0.20, random_state=state);
                reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
                fit!(reg, X_train, y_train)
                z[(i*itr+j),1] = leaf
                z[(i*itr+j),2] = tree
                z[(i*itr+j),3] = i
                z[(i*itr+j),4] = state
                z[(i*itr+j),5] = score(reg, X_train, y_train)
                z[(i*itr+j),6] = score(reg, X_test, y_test)
            end
            println("End of $i FP type (see descriptors.xml)")
        end    
        z_df = DataFrame(leaves = z[:,1], trees = z[:,2], FP_type = z[:,3], state=z[:,4], accuracy_train = z[:,5], accuracy_test = z[:,6])
        z_df_sorted = sort(z_df, :accuracy_test, rev=true)
        return z_df_sorted
    end

    z_df_sorted_plus = optim(data_plus[:,3],+1)
    CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_plus.csv", z_df_sorted_plus)

    opt_res_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_minus.csv", DataFrame)
    opt_res_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\General_FP_optimisation_results_plus.csv", DataFrame)

    scatter(opt_res_minus[:,3], opt_res_minus[:,end])

## Type-12 fingerprints optimization ##
function optim_type12(ESI; itr::Int=30)
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(50:50:150),collect(200:200:1000))
    state_r = collect(1:3)
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    results = zeros(itr,5)
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    variables = Matrix(hcat(FP[!,:pH_aq],FP[!,9:end]))
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(state_r)
            MaxFeat = Int64(ceil(size(variables,2)/3))

            function split_classes(ESI, classes; random_state::Int=state, split_size::Float64=0.2)
                if ESI == -1
                    ESI_name = "neg"
                elseif ESI == 1
                    ESI_name = "pos"
                else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
                end
                FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
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
            # Stratified splitting
            classes = unique(FP[:,:INCHIKEY])
            train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=state)

            test_set_indices = findall(x -> x in test_set_inchikeys, FP[:,:INCHIKEY])
            train_set_indices = findall(x -> x in train_set_inchikeys, FP[:,:INCHIKEY])

            X_train = variables[train_set_indices,:]
            X_test = variables[test_set_indices,:]
            y_train =  FP[train_set_indices,:logIE]
            y_test = FP[test_set_indices,:logIE]
        ## Regression ##
            reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
            fit!(reg, X_train, y_train)
            results[j,1] = leaf
            results[j,2] = tree
            results[j,3] = state
            results[j,4] = score(reg, X_train, y_train)
            results[j,5] = mean(cross_val_score(reg, X_train, y_train, cv=10))
            println("End of $j / $itr iterations")
        end
    results_df = DataFrame(leaves = results[:,1], trees = results[:,2], state=results[:,3], accuracy_train = results[:,4], accuracy_test = results[:,5])
    results_df_sorted = sort(results_df, :accuracy_test, rev=true)
    return results_df_sorted
end

z = optim_type12(-1,50)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Type12_Optimisation_results_minus.csv", z)

x = optim_type12(+1, itr=50)
CSV.write("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\Type12_Optimisation_results_plus_CV.csv", x)

x
scatter(x[:,:trees],x[:,:accuracy_train], label = "Training set", xlabel = "No. of trees", ylabel="Accuracy (R2)", legend=:best, ylims=(0.5,1))
scatter!(x[:,:trees],x[:,:accuracy_test], label = "Cross-validation")
savefig("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\RF_trees.png")

scatter(x[:,:leaves],x[:,:accuracy_train], label = "Training set", xlabel = "Minimum samples per leaf", ylabel="Accuracy (R2)", legend=:best, ylims=(0.5,1))
scatter!(x[:,:leaves],x[:,:accuracy_test], label = "Cross-validation")
savefig("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\RF_leaves.png")

scatter(x[:,:state],x[:,:accuracy_train], label = "Training set", xlabel = "Random state", ylabel="Accuracy (R2)", legend=:best, ylims=(0.5,1))
scatter!(x[:,:state],x[:,:accuracy_test], label = "Cross-validation")
savefig("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\RF_state.png")

state=1
ESI=+1
## Importance for Padel-12 ##
function parameter(ESI; allowplots=false, allowsave=false, showph=false)
    if ESI == -1
        ESI_name = "neg"
        ESI_symbol = "-"
        output = data_minus[:,3]
        reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=4, max_features=(Int64(ceil(size(FP1,2)/3))), n_jobs=-1, oob_score =true, random_state=2)
    elseif ESI == 1
        ESI_name = "pos"
        ESI_symbol = "+"
        state = 1
        reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=4, max_features=261, n_jobs=-1, oob_score =true, random_state=state)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    variables_df = hcat(FP[!,:pH_aq],FP[!,9:end])
    variables = Matrix(hcat(FP[!,:pH_aq],FP[!,9:end]))

    function split_classes(ESI, classes; random_state::Int=state, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
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
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end
    # Stratified splitting
    classes = unique(FP[:,:INCHIKEY])
    train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=state)

    test_set_indices = findall(x -> x in test_set_inchikeys, FP[:,:INCHIKEY])
    train_set_indices = findall(x -> x in train_set_inchikeys, FP[:,:INCHIKEY])

    X_train = variables[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_train =  FP[train_set_indices,:logIE]
    y_test = FP[test_set_indices,:logIE]
    ## Regression ##
    fit!(reg, X_train, y_train)
    importance = 100 .* sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]
    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = score(reg, X_train, y_train)   # Train set accuracy
    z3 = score(reg, X_test, y_test)      # Test set accuracy
    z4 = predict(reg,X_train)     # y_hat_train
    z5 = predict(reg,X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual
    
    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title="ESI$(ESI_symbol) IEs from FP", color = :magenta, xlabel="Experimental log(IE)", ylabel = "Predicted log(IE)")
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
            savefig("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\FP_RF_Regression_$ESI_name.png")
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
            savefig("C:\\Users\\alex_\\Desktop\\Master\\Advanced chemometrics and statistics\\Project\\FP_RF_Residuals_$ESI_name.png")
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
                #savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                #savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
end


importance_minus, accuracy_tr_minus, accuracy_te_minus, y_hat_train_minus, y_hat_test_minus, res_train_minus, res_test_minus = parameter(-1, allowplots=true, allowsave=true)
importance_plus, accuracy_tr_plus, accuracy_te_plus, y_hat_train_plus, y_hat_test_plus, res_train_plus, res_test_plus = parameter(+1, allowplots=true, allowsave=true)



## Cross Validation
CV_mean = mean(cross_val_score(reg, X_train, y_train, cv=10))

importance_plus[1:10]

# 
FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)

####### Meeting notes #######
## DONE ##
# Morgan Fingerprints (if time)
# Positive mode
# Residuals + plots

## IN PROGRESS ##
# Importance --> WRITE IN TABLE

## TO DO ##
# Inter-Correlation
# Find abnormalities to show model strengths and weaknesses
# Different types of models (CATBOOST, XgBoost)
# CATBOOST -> Learning rate, LU1_reg, LU2_reg
# XgBoost -> Learning rate & RandomForestRegressor

## CNL ##
# InChI keys creation
# Talk with Denice