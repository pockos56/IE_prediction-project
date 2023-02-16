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

@sk_import ensemble: RandomForestClassifier
rdk = pyimport("rdkit.Chem")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")


#
# Create vector of pH-dependent and pH independent compounds (0=Unknown, 1=Unstable, 2=Stable)
function diff_classes(ESI; threshold=0.1)

    if ESI == -1
        ESI_name = "minus"
    elseif ESI == 1
        ESI_name = "plus"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    #load files
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_$(ESI_name)_12.csv", DataFrame)
    FP1 = hcat(FP[!,:pH_aq],FP[!,8:end])
    data = FP[:,[:name,:pH_aq, :logIE]]

    comp_names = unique(data[:,:name])
    pH_stability = Int64.(zeros(length(data[:,1])))
    for i = 1:length(comp_names)
        data_temp = data[data[:,1] .== comp_names[i],:]
        if length(unique(data_temp[:,2])) > 1
            if maximum(data_temp[:,:logIE]) - minimum(data_temp[:,:logIE]) > threshold
                pH_stability[data[:,1] .== comp_names[i]] .= 1 # Set 1 to unstable compounds
            elseif maximum(data_temp[:,:logIE]) - minimum(data_temp[:,:logIE]) <= threshold
                pH_stability[data[:,1] .== comp_names[i]] .= 2 # Set 2 to stable compounds
            end
        end
    end
    unknown = findall(x -> x .== 0, pH_stability)
    unstable = findall(x -> x .== 1, pH_stability)
    stable = findall(x -> x .== 2, pH_stability)
    return pH_stability, unknown, unstable, stable
end


classes_minus, unknown_minus, unstable_minus, stable_minus = diff_classes(-1,threshold=0.7)
classes_plus, unknown_plus, unstable_plus, stable_plus = diff_classes(+1,threshold=0.7);


#Names of the compounds that are stable
minus_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12.csv", DataFrame)
plus_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12.csv", DataFrame)
sort(minus_raw[stable_minus,:])
plus_raw[stable_plus,:]
# Create classifier

X_train, X_test, y_train, y_test = train_test_split(Matrix(FP1), FP[!,:logIE], test_size=0.20, random_state=2);
reg = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, max_features=0, n_jobs=-1, oob_score =true, random_state=2)
fit!(reg, X_train, y_train)













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