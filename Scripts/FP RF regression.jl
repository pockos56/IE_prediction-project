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
using LinearAlgebra
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


ESI=-1
## Importance for Padel-12 ##
function Stratified_FP_model(ESI; allowplots=false, allowsave=false, showph=false, random_seed::Int = 2539)

    function split_classes(ESI, classes; random_state::Int=1312, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
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
        
        function cityblock_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
                #println(j)
            end
            return lev
        end
    
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)
        #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
        
        inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))
    
        return inchi_train, inchi_test
    end


    if ESI == -1
        ESI_name = "neg"
        output = data_minus[:,:logIE]
    elseif ESI == 1
        ESI_name = "pos"
        output = data_plus[:,:logIE]
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,8:end])

        # Stratified splitting
        classes = unique(FP[:,:INCHIKEY])
        train_set_inchikeys,test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

        test_set_indices = findall(x -> x in test_set_inchikeys, FP[:,:INCHIKEY])
        train_set_indices = findall(x -> x in train_set_inchikeys, FP[:,:INCHIKEY])

        X_train = FP1[train_set_indices,:]
        X_test = FP1[test_set_indices,:]
        y_train =  FP[train_set_indices,:logIE]
        y_test = FP[test_set_indices,:logIE]
    
    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_features=(Int64(ceil(size(Matrix(FP1),2)/3))), n_jobs=-1, oob_score =true, random_state=2)
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
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\RF_Regression_M2M4_$ESI_name.png")
        end

        plot2 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\RF_Residuals_M2M4_$ESI_name.png")
        end
        display(plot1)
        display(plot2)
        if showph == true
            X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), Matrix(FP[!,[:pH_aq,:logIE]]), test_size=0.20, random_state=2);
            reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, max_features=(Int64(ceil(size(Matrix(FP1),2)/3))), n_jobs=-1, oob_score =true, random_state=2)
            fit!(reg, X_train, y_train[:,2])
            z4 = predict(reg,X_train)     # y_hat_train
            z5 = predict(reg,X_test)   # y_hat_test  
            z6 = z4 - y_train[:,2]      # Train set residual
            z7 = z5 - y_test[:,2]        # Test set residual
           
            
            plot_pH = scatter(y_train[:,2],z4,label="Training set", legend=:best, title = "ESI $(ESI_name)- IEs from FP", markershape = :circle, marker_z = y_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test[:,2],z5,label="Test set", marker_z = y_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))],[minimum(vcat(y_train[:,2],y_test[:,2])),maximum(vcat(y_train[:,2],y_test[:,2]))], label="1:1 line",width=2)
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\RF_Regression_pHcolor_M2M4_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train[:,2],z6,label="Training set", legend=:best, title = "ESI $(ESI_name)- Regression residuals",markershape=:circle, marker_z=y_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test[:,2],z7, markershape=:rect,marker_z=y_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test[:,2],y_train[:,2])),maximum(vcat(y_test[:,2],y_train[:,2]))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\RF_Residuals_pHcolor_M2M4_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
return z1,z2,z3,z4,z5,z6,z7
end

importance_minus, accuracy_tr_minus, accuracy_te_minus, y_hat_train_minus, y_hat_test_minus, res_train_minus, res_test_minus = parameter(-1, allowplots=true, allowsave=true,showph=true)
importance_plus, accuracy_tr_plus, accuracy_te_plus, y_hat_train_plus, y_hat_test_plus, res_train_plus, res_test_plus = parameter(+1, allowplots=true, allowsave=true, showph=true)
importance_plus
importance_minus


## pH distribution
histogram(data_minus[:,:pH_aq], bins=20 ,label = "ESI -",xlims=(0,14))
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\pH distribution ESI-.png")
histogram(data_plus[:,:pH_aq], bins=20 ,label = "ESI +",xlims=(0,14))
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\pH distribution ESI+.png")

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

## 
#Run the test set with multiple pHs multiple iterations, outputting the range of IEs and the optimal pH to be used.
# Correlation to understand the logic behind the most important feature_importances
# Check with the Cat model for the correlation to see if sth is considered there to explain the smaller residual for extreme IEs.