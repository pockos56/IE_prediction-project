using CSV, DataFrames, PyCall, Conda, ProgressBars, Statistics, LinearAlgebra, Plots, Random
#using Pkg
#Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")
using IE_prediction

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

# Leverage calculation function
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
# Sample every nth INCHIKEY sorted by leverage 
randomness = 1
expa_comps = (sort(x, "leverage")[unique(Int.(round.(collect(randomness:25376/835:size(x,1))))),:])[:,1]
expa_spectra = Internal_pos_H[findall(x->x in expa_comps, Internal_pos_H[:,"INCHIKEY"]),:]

# Predict IEs
# Choose a pH from the training set pH distribution
FP_training_set = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\FP_model_training set.csv", DataFrame)
CNL_IE_unified_zero = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero.csv", DataFrame)
histogram(FP_training_set[:,"pH.aq."], label="FP train set pH", bins=50)
histogram(CNL_IE_unified_zero[:,"pH_aq"], label="CNL-IE original pH", bins=50)

# The two distributions are roughly the same, as expected. We choose FP_training_set pH distribution
# We have 4570 original pH values and need 20988 output values
pH_vec_original = FP_training_set[:,"pH.aq."]
pH_vec_1 = pH_vec_original[randperm(MersenneTwister(1), size(pH_vec_original,1))]
pH_vec_2 = pH_vec_original[randperm(MersenneTwister(2), size(pH_vec_original,1))]
pH_vec_3 = pH_vec_original[randperm(MersenneTwister(3), size(pH_vec_original,1))]
pH_vec_4 = pH_vec_original[randperm(MersenneTwister(4), size(pH_vec_original,1))]
pH_vec_1_4 = vcat(vcat(pH_vec_1,pH_vec_2),vcat(pH_vec_3,pH_vec_4))
pH_vec_5 = pH_vec_original[randperm(MersenneTwister(5), size(pH_vec_original,1))][1:(size(expa_spectra,1)-length(pH_vec_1_4))]
pH_vec = vcat(pH_vec_1_4,pH_vec_5)
histogram(FP_training_set[:,"pH.aq."], label="FP train set pH", bins=50, alpha=0.7)
histogram!(pH_vec, label="Expansion pH", bins=50, alpha=0.7)
# As shown in the histogram, the pH distribution of the expansion set is the same as the pH distribution of the FP training set
# Last let's shuffle expa_spectra before inserting the pH_vec as an extra step of (reproducible) randomness
expa_spectra_wpH = expa_spectra[randperm(MersenneTwister(1312), size(expa_spectra,1)),:]
insertcols!(expa_spectra_wpH, 6, :pH_aq =>pH_vec)
logIE_vec = ones(size(expa_spectra_wpH,1)) .* -Inf
insertcols!(expa_spectra_wpH, 7, :logIE =>logIE_vec)
for i in ProgressBar(1:size(expa_spectra_wpH,1))
    try
    expa_spectra_wpH[i,"logIE"] = ulogIE_from_InChIKey(String(expa_spectra_wpH[i,"INCHIKEY"]), expa_spectra_wpH[i,"pH_aq"])
    catch
        error("Comp $i not calculated.")
        break
    end
end

# Find the ones that couldn't be predicted
not_calculated = findall(x->x.==-Inf, expa_spectra_wpH[:,"logIE"])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\Expansion.csv", expa_spectra_wpH)

# Load data
CNL_IE_unified_zero = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero.csv", DataFrame)
expansion = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\Expansion.csv", DataFrame)

# Checking that everything is according to the requirements for semi-supervision
size(CNL_IE_unified_zero,1)     # 227.5k spectra
size(expansion,1)               # 20.9k spectra
unique(CNL_IE_unified_zero[:,"INCHIKEY"])   # 835 unique INCHIKEYs
unique(expansion[:,"INCHIKEY"])             # 835 unique INCHIKEYs
findall(x -> x in unique(CNL_IE_unified_zero[:,"INCHIKEY"]), unique(expansion[:,"INCHIKEY"]))   # No inchikey overlap

# Prepare expansion set to CNL binary format

# Combine data
names(CNL_IE_unified_zero)
names(expansion)