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
function optim_type12(ESI, iterations=50)
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(50:50:400),collect(500:100:1000))
    itr = iterations
    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    z = zeros(itr,5)
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_$(ESI_name)_12.csv", DataFrame)
        FP1 = Matrix(hcat(FP[!,:pH_aq],FP[!,8:end]))
        for j = 1:itr
            leaf = rand(leaf_r)
            tree = rand(tree_r)
            state = rand(1:3)
            MaxFeat = Int64(ceil(size(FP1,2)/3))
        ## Regression ##
            X_train, X_test, y_train, y_test = train_test_split(FP1, FP[!,:logIE], test_size=0.20, random_state=state);
            reg = RandomForestRegressor(n_estimators=tree, min_samples_leaf=leaf, max_features=MaxFeat, n_jobs=-1, oob_score =true, random_state=state)
            fit!(reg, X_train, y_train)
            z[j,1] = leaf
            z[j,2] = tree
            z[j,3] = state
            z[j,4] = score(reg, X_train, y_train)
            z[j,5] = score(reg, X_test, y_test)
            println("End of $j / $itr iterations")
        end
    z_df = DataFrame(leaves = z[:,1], trees = z[:,2], state=z[:,3], accuracy_train = z[:,4], accuracy_test = z[:,5])
    z_df_sorted = sort(z_df, :accuracy_test, rev=true)
    return z_df_sorted
end

z_df_sorted_minus = optim_morgan(data_minus[:,3],-1)
z_df_sorted_plus = optim_morgan(data_plus[:,3],+1)

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Morgan_FP_optimisation_results_minus.csv", z_df_sorted_minus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Morgan_FP_optimisation_results_plus.csv", z_df_sorted_plus)

z = optim_type12(-1,20)
## Importance for Padel-12 ##
function parameter(ESI; allowplots=false, allowsave=false)
    if ESI == -1
        ESI_name = "minus"
        output = data_minus[:,3]
    elseif ESI == 1
        ESI_name = "plus"
        output = data_plus[:,3]
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_$(ESI_name)_12.csv", DataFrame)[:,2:end]
    X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), output, test_size=0.20, random_state=2);
    reg = RandomForestRegressor(n_estimators=600, min_samples_leaf=4, max_features=(Int64(ceil(size(FP1,2)/3))), n_jobs=-1, oob_score =true, random_state=2)
    fit!(reg, X_train, y_train)
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
        if allowsave ==true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\RF_Regression_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave ==true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\RF_Residuals_$ESI_name.png")
        end
        display(plot1)
        display(plot2)
    end
return z1,z2,z3,z4,z5,z6,z7
end


importance_minus, accuracy_tr_minus, accuracy_te_minus, y_hat_train_minus, y_hat_test_minus, res_train_minus, res_test_minus = parameter(-1, allowplots=true, allowsave=true)
importance_plus, accuracy_tr_plus, accuracy_te_plus, y_hat_train_plus, y_hat_test_plus, res_train_plus, res_test_plus = parameter(+1, allowplots=true, allowsave=true)

FP1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_minus_12.csv", DataFrame)[:,2:end]
sort(unique(Matrix(FP1)[:,:]))
importance_minus
importance_plus


## Cross Validation
CV = cross_val_score(reg, X_train, y_train, cv=5)
CV_mean = mean(CV)

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