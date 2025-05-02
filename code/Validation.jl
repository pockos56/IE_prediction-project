using CSV, DataFrames, ProgressBars, IE_prediction, PyCall
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

#using Pkg
#Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")
# Load files
harry = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\data_for_Alex_from_Harry.csv",DataFrame)
helen= CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\data_for_Alex_from_Helen.csv",DataFrame)
data_raw = vcat(harry,helen)
#CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Validation_set_inchikeys.csv", DataFrame(INCHIKEY = data_raw.INCHIKEY))
data = deepcopy(data_raw)

# Check if the validation set INCHIKEYs are present in the training sets
data_raw.INCHIKEY .= "To Be Calculated"
for i in ProgressBar(1:size(data_raw,1)) 
    cids = pcp.get_compounds(data_raw[i,"inchi"], "inchi")
    shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
    data_raw[i,"INCHIKEY"] = shortest_cid.inchikey
end
data_raw[:,Not(:inchi,:SMILES,:measmass,:ion_mode)]

Common_INCHIKEYs_FP_min = data_raw[findall(x->x in unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_min.csv", DataFrame).INCHIKEY), data_raw.INCHIKEY),Not(:inchi,:measmass,:ion_mode,:pH_aq)]
Common_INCHIKEYs_FP_mean = data_raw[findall(x->x in unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_mean.csv", DataFrame).INCHIKEY), data_raw.INCHIKEY),Not(:inchi,:measmass,:ion_mode,:pH_aq)]
Common_INCHIKEYs_FP_max = data_raw[findall(x->x in unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_max.csv", DataFrame).INCHIKEY), data_raw.INCHIKEY),Not(:inchi,:measmass,:ion_mode,:pH_aq)]
Common_INCHIKEYs_CNL_min = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_min.csv", DataFrame)[findall(x->x in data_raw.INCHIKEY, unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_min.csv", DataFrame).INCHIKEY)),:]
Common_INCHIKEYs_CNL_mean = data_raw[findall(x->x in unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_mean.csv", DataFrame).INCHIKEY), data_raw.INCHIKEY),Not(:inchi,:measmass,:ion_mode,:pH_aq)]
Common_INCHIKEYs_CNL_max = data_raw[findall(x->x in unique(CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_max.csv", DataFrame).INCHIKEY), data_raw.INCHIKEY),Not(:inchi,:measmass,:ion_mode,:pH_aq)]

# First, predict the structure-based IE
# -- from SMILES
data_raw.logIE_from_SMILES_min .= 0.0
for i in ProgressBar(1:size(data_raw,1)) 
    data_raw[i,"logIE_from_SMILES_min"] = logIE_from_SMILES(data_raw[i,"SMILES"], data_raw[i,"pH_aq"], "min")
end
# -- from inchi
data_raw.logIE_from_inchi_min .= 0.0
for i in ProgressBar(1:size(data_raw,1))
    cids = pcp.get_compounds(data_raw[i,"inchi"], "inchi")
    shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
    data_raw[i,"logIE_from_inchi_min"] = logIE_from_SMILES(shortest_cid.isomeric_smiles, data_raw[i,"pH_aq"], "min")
end
# -- from name
data_raw.logIE_from_name_min .= 0.0
for i in ProgressBar(1:size(data_raw,1))
    cids = pcp.get_compounds(data_raw[i,"name"], "name")
    shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
    data_raw[i,"logIE_from_name_min"] = logIE_from_SMILES(shortest_cid.isomeric_smiles, data_raw[i,"pH_aq"], "min")
end
data[:,Not(:inchi,:SMILES,:measmass,:ion_mode)]

# Calculate from inchi 
data.logIE_from_inchi_min .= 0.0
data.logIE_from_inchi_mean .= 0.0
data.logIE_from_inchi_max .= 0.0
for i in ProgressBar(1:size(data,1))
    cids = pcp.get_compounds(data[i,"inchi"], "inchi")
    shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
    data[i,"logIE_from_inchi_min"] = logIE_from_SMILES(shortest_cid.isomeric_smiles, data[i,"pH_aq"], "min")
    data[i,"logIE_from_inchi_mean"] = logIE_from_SMILES(shortest_cid.isomeric_smiles, data[i,"pH_aq"], "mean")
    data[i,"logIE_from_inchi_max"] = logIE_from_SMILES(shortest_cid.isomeric_smiles, data[i,"pH_aq"], "max")
end

# Second, calculate the CNL IE
data.logIE_from_cnl_min .= 0.0
data.logIE_from_cnl_mean .= 0.0
data.logIE_from_cnl_max .= 0.0
for i in ProgressBar(1:size(data,1))
    mz_vec = eval(Meta.parse(replace(data[i,:measmass], r"Any(\[.*?\])" => s"\1")))
    data[i,"logIE_from_cnl_min"] = logIE_from_CNLs(mz_vec, data[i,"precursor_ion"], data[i,"pH_aq"], "min")
    data[i,"logIE_from_cnl_mean"] = logIE_from_CNLs(mz_vec, data[i,"precursor_ion"], data[i,"pH_aq"], "mean")
    data[i,"logIE_from_cnl_max"] = logIE_from_CNLs(mz_vec, data[i,"precursor_ion"], data[i,"pH_aq"], "max")
end
# Results
results = data[:,Not(:inchi,:ion_mode,:pH_aq,:measmass,:precursor_ion)]
results = data[:,Not(:inchi)]

unique(results[:,"name"])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Predicted_IEs_validation_set.csv", data)
# Plots
using Plots
mode = "mean"
histogram(data[:,"logIE_from_inchi_$mode"], bins=20, alpha=0.7, title="Validation set - $mode", label="Structure-based prediction", xlabel="logIE",legend=:topleft, xlims=(0.8,5))
histogram!(data[:,"logIE_from_cnl_$mode"], bins=15, alpha=0.7,label="MS-based prediction")