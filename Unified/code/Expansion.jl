using CSV, DataFrames, PyCall, Conda, ProgressBars, Statistics, LinearAlgebra, Plots
using ScikitLearn.CrossValidation: train_test_split
using ScikitLearn

# Load data
Internal_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)
data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", DataFrame)
Internal_raw[:,["INCHIKEY", "INCHI", "SMILES", "PUBCHEM"]]
sort(unique(Internal_raw[:,"PUBCHEM"]))
# Filtering
Internal_pos = vcat(Internal_raw[Vector(Internal_raw[:,:ION_MODE]).=="P",:],Internal_raw[Vector(Internal_raw[:,:ION_MODE]).=="POSITIVE",:])
Internal_pos_H = Internal_pos[((Internal_pos[:,:PRECURSOR_ION] - Internal_pos[:,:EXACT_MASS]) .< 1.01),:]
unique_comps = combine(groupby(Internal_pos_H, "INCHIKEY")) do group first(group, 1) end

# Compute the PubChem fingerprints for these 26.8k INCHIKEYs
rep = unique_comps[4:6,:]
function from_smiles_vector2(rep::DataFrame)
    fp_padel=DataFrame()
    cids = pcp.get_compounds(rep[1,"INCHIKEY"], "inchikey")
    try
        shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
        fp_padel = hcat(DataFrame(rep[1,:]), DataFrame(pd.from_smiles(shortest_cid.isomeric_smiles,fingerprints=true, descriptors=false)))    
    catch
    end
    for i in ProgressBar(2:size(rep,1))
        try
            cids = pcp.get_compounds(rep[i,"INCHIKEY"], "inchikey")
            shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
            fp_padel_temp = hcat(DataFrame(rep[i,:]), DataFrame(pd.from_smiles(shortest_cid.isomeric_smiles,fingerprints=true, descriptors=false)))
            fp_padel = append!(fp_padel,fp_padel_temp)
        catch
            continue
        end
    end
    return fp_padel
end
test = from_smiles_vector2(unique_comps)

# Save fingerprints

# Load fingerprints

# Leverage calculation
function leverage_dist(training_set_FPs, expansion_set_FPs)
    training_set = pinv(transpose(training_set_FPs) * training_set_FPs)

    lev = zeros(size(expansion_set_FPs,1))
    for j in ProgressBar(1:size(expansion_set_FPs,1)) 
        lev[j] = transpose(expansion_set_FPs[j,:]) * training_set * expansion_set_FPs[j,:]
    end
    return lev
end

# The CNL_IE_unified is the training set for the CNL model
CNL_IE_unified_zero = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero.csv", DataFrame)
# It has 835 unique compounds
unique_comps_CNLIE = combine(groupby(CNL_IE_unified_zero, "INCHIKEY")) do group first(group, 1) end
# The Internal_FPs_dict contains the PubChem FPs for all Internal_pos_H compounds
Internal_FPs_dict = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Internal_pubchem_FPs_dict.csv", DataFrame)
#We should exclude the ones that are present in the current (prior to expansion) CNL-IE dataset
same_comps_ind = findall(x->x in unique_comps_CNLIE[:,"INCHIKEY"], Internal_FPs_dict[:,"INCHIKEY"])
Internal_FPs_dict_filtered = deepcopy(Internal_FPs_dict)
deleteat!(Internal_FPs_dict_filtered, same_comps_ind)
same_comps_ind_verify = findall(x->x in unique_comps_CNLIE[:,"INCHIKEY"], Internal_FPs_dict_filtered[:,"INCHIKEY"])     #Returns nothing. The filtered dict has different comps than the training set comps
# The FP_training_set contains the PubChem FPs for all comps used in the optimized FP model
FP_training_set = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_model_training set.csv", DataFrame)
same_comps_ind = findall(x->x in unique(FP_training_set[:,"INCHIKEY"]), Internal_FPs_dict_filtered[:,"INCHIKEY"])
deleteat!(Internal_FPs_dict_filtered, same_comps_ind)
# Calculate the leverage for these comps
Internal_leverages = leverage_dist(Matrix(FP_training_set[:,10:end]), Matrix(Internal_FPs_dict_filtered[:,2:end]))
insertcols!(Internal_FPs_dict_filtered, 2, :leverage =>Internal_leverages)
# Decide the compounds for the expansion
# CNL_IE has 835 unique INCHIKEYs in 227519 spectra
unique(CNL_IE_unified_zero[:,"INCHIKEY"])
CNL_IE_unified_zero
# Internal_FPs_dict_filtered has 25376 unique INCHIKEYs in 641682 spectra
Internal_FPs_dict_filtered
Internal_pos_H[:,:INCHIKEY]
        #findall(x->x in Internal_FPs_dict_filtered[:,"INCHIKEY"], Internal_pos_H[:,"INCHIKEY"])
# First pick 835 comps with stratification and get all spectra (which should be less than 227k)
Internal_FPs_dict_filtered
x = Internal_FPs_dict_filtered[:,["INCHIKEY","leverage"]]
# First try: Sample with train_test_split stratification (Labeling is poor)
a1, a2 = train_test_split(Matrix(x), test_size=835, random_state=1, stratify = round.(Vector(x[:,2]),digits = 1))
# Second try: Sample every nth INCHIKEY sorted by leverage 
randomness = 1
exp_comps = (sort(x, "leverage")[unique(Int.(round.(collect(randomness:25376/835:size(x,1))))),:])[:,1]
exp_spectra = Internal_pos_H[findall(x->x in exp_comps, Internal_pos_H[:,"INCHIKEY"]),:]



