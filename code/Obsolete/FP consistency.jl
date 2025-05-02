using CSV, DataFrames, PyCall, Conda, PaDELjl, ProgressBars, Random
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

# Loading data
data = CSV.read("/home/emcms/Data/Data Alex/IE_prediction/unified/data/filtered_data.csv", DataFrame)
#data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\filtered_data.csv", DataFrame)

smiles_ = data[:,:SMILES_cor]
smiles_short = smiles_[randperm(MersenneTwister(69),length(smiles_))[1:500]]
# PadelJL fingerprints
fp_vec1 = from_smiles_batch(smiles_short, 25, fingerprints=true, descriptors=false)
fp_vec2 = from_smiles_batch(smiles_short, 25, fingerprints=true, descriptors=false)
fp_vec3 = from_smiles_batch(smiles_short, 5, fingerprints=true, descriptors=false)
# padelpy fingerprints
function from_smiles_vector(canonical_smiles::Vector)
    fp_padel = DataFrame(pd.from_smiles(canonical_smiles[1],fingerprints=true, descriptors=false))
    for i in ProgressBar(2:length(canonical_smiles))
        fp_padel = append!(fp_padel, DataFrame(pd.from_smiles(canonical_smiles[i],fingerprints=true, descriptors=false)))
    end
    return fp_padel
end
fp_padel1 = from_smiles_vector(smiles_short)
fp_padel2 = from_smiles_vector(smiles_short)
fp_padel3 = from_smiles_vector(smiles_short)
# Checking for inconsistencies
function inconsistency_discovery(fp_vec1, fp_vec2, fp_vec3, used_smiles)
    falses_12 = findall(x -> x.==false,Matrix(fp_vec1 .== fp_vec2))
    falses_13 = findall(x -> x.==false,Matrix(fp_vec1 .== fp_vec3))
    falses_23 = findall(x -> x.==false,Matrix(fp_vec2 .== fp_vec3))
    falses = unique(vcat(vcat(falses_12,falses_13),falses_23))
    as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))
    falses_cols = unique(as_ints(falses)[2,:])
    colnames = names(fp_vec1[:,falses_cols])
    falses_comps = used_smiles[unique(as_ints(falses)[1,:])]
    return colnames, falses_comps
end
colnames_jl, false_comps_jl = inconsistency_discovery(fp_vec1, fp_vec2, fp_vec3, smiles_short)
colnames_py, false_comps_py = inconsistency_discovery(fp_padel1, fp_padel2, fp_padel3, smiles_short)
# Conclusion #
# The PadelJL fingerprints calculation was tested for Pubchem fingerprints, with batch size from 50 to 5 in triplicates. In all tests the triplicates were found to be wrongly calculated in at least one of the calculations. This shows that PadelJL cannot be used for robust fingerprint calculation. Conda was updated but the results were the same. 
# Same 500 smiles were calculated using padelpy via Conda. No inconsistencies were detected. 
