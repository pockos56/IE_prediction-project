using PaDELjl
using CSV, DataFrames, ProgressBars
Internal = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)
smiles_ = unique(convert.(String, Internal[:,:SMILES]))[1:1000]

fp_vec1 = DataFrame()
fp_vec2 = DataFrame()
fp_vec3 = DataFrame()
for i in ProgressBar(1:5:length(smiles_)) 
    fp_vec1 = append!(fp_vec1, from_smiles(smiles_[i:i+4], fingerprints=true, descriptors=false))
    fp_vec2 = append!(fp_vec2, from_smiles(smiles_[i:i+4], fingerprints=true, descriptors=false))
    fp_vec3 = append!(fp_vec3, from_smiles(smiles_[i:i+4], fingerprints=true, descriptors=false))
end

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC1.csv", fp_vec1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC2.csv", fp_vec2)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC3.csv", fp_vec3)

fp_apc1 = fp_vec1
fp_apc2 = fp_vec2
fp_apc3 = fp_vec3
#Comparison
fp1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_1.csv", DataFrame)
fp2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_2.csv", DataFrame)
fp_apc1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC1.csv", DataFrame)
fp_apc2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC2.csv", DataFrame)
fp_apc3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_APC3.csv", DataFrame)

falses = findall(x -> x.==false,Matrix(fp1 .== fp2))
falses_12 = findall(x -> x.==false,Matrix(fp_apc1 .== fp_apc2))
falses_13 = findall(x -> x.==false,Matrix(fp_apc1 .== fp_apc3))
falses_23 = findall(x -> x.==false,Matrix(fp_apc2 .== fp_apc3))

as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

falses_matrix = as_ints(falses)
falses_12_matrix = as_ints(falses_12)
falses_13_matrix = as_ints(falses_13)
falses_23_matrix = as_ints(falses_23)

falses_cols = unique(as_ints(falses)[2,:])
falses_cols_12 = unique(as_ints(falses_12)[2,:])
falses_cols_13 = unique(as_ints(falses_13)[2,:])
falses_cols_23 = unique(as_ints(falses_23)[2,:])

# Names of the unstable fingerprints
names(fp1[:,falses_cols])
fp1=[]
names(fp_apc1[:,falses_cols_12])
names(fp_apc2[:,falses_cols_13])
names(fp_apc3[:,falses_cols_23])
all_unstable_colnames = unique(reduce(vcat,(names(fp_apc1[:,falses_cols_12]),names(fp_apc2[:,falses_cols_13]),names(fp_apc3[:,falses_cols_23]))))
#CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\Unstable fingerprints.csv", DataFrame(all_unstable_colnames[:,:],:auto))

falses_comps = smiles_[unique(as_ints(falses)[1,:])]
falses_comps_12 = smiles_[unique(as_ints(falses_12)[1,:])]
falses_comps_13 = smiles_[unique(as_ints(falses_13)[1,:])]
falses_comps_23 = smiles_[unique(as_ints(falses_23)[1,:])]
all_unstable_comps = unique(vcat(vcat(falses_comps_12),vcat(falses_comps_13,falses_comps_23)))

# Comp 1719     # Pubchem # Inconsistency with aromatic rings and S atoms
names(fp_apc1[:,as_ints(falses_12)[:,as_ints(falses_12)[1,:].==1719][2,:]])
# "APC2D1_C_C" issue # APC2D1_C_C #All compounds have S atoms
falses_cols_23[findall(x->x.=="APC2D1_C_C", names(fp_apc3[:,falses_cols_23]))[1]]       # 6215th column
smiles_[as_ints(falses_23)[:,as_ints(falses_23)[2,:].==6215][1,:]]

# General descriptors calculation
using BSON
using Statistics
using DataFrames
using CSV
using PyCall
using Conda
using Plots
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
alc = pyimport("rdkit.Chem.AllChem")

iterations = 3
smiles = all_unstable_comps
fingerprint = DataFrame(pd.from_smiles(smiles,fingerprints=true, descriptors=false))
for i = 2:iterations
    fingerprint = vcat(fingerprint, DataFrame(pd.from_smiles(smiles,fingerprints=true, descriptors=false)))
end

for i = 2:iterations
    falses = findall(x -> x.==false,Vector(Vector(fingerprint[1,:]) .== Vector(fingerprint[i,:])))
    if isempty(falses) == false
        return falses
    end
end

using PaDELjl

smiles_batch = [smiles, smiles, smiles, smiles, smiles]
smiles_batch = all_unstable_comps
fingerprint1 = from_smiles_batch(smiles_batch, 5, fingerprints=true, descriptors=false)
fingerprint2 = from_smiles_batch(smiles_batch, 10, fingerprints=true, descriptors=false)
fingerprint3 = from_smiles_batch(smiles_batch, 25, fingerprints=true, descriptors=false)
fingerprint4 = from_smiles_batch(smiles_batch, 25, fingerprints=true, descriptors=false)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_all_batch5_badcompsonly.csv", fingerprint1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_all_batch10_badcompsonly.csv", fingerprint2)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP recalculation\\FP_all_batch25_badcompsonly.csv", fingerprint3)

falses_12 = findall(x -> x .== 0,Matrix(fingerprint1 .== fingerprint2))
falses_23 = findall(x -> x .== 0,Matrix(fingerprint2 .== fingerprint3))
falses_13 = findall(x -> x .== 0,Matrix(fingerprint1 .== fingerprint3))

as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

#falses_matrix = as_ints(falses)
falses_12_matrix = as_ints(falses_12)
falses_13_matrix = as_ints(falses_13)
falses_23_matrix = as_ints(falses_23)

falses_cols = unique(as_ints(falses)[2,:])
falses_cols_12 = unique(as_ints(falses_12)[2,:])
falses_cols_13 = unique(as_ints(falses_13)[2,:])
falses_cols_23 = unique(as_ints(falses_23)[2,:])

# Names of the unstable fingerprints
names(fp1[:,falses_cols])
names1 = names(fingerprint1[:,falses_cols_12])
names2 = names(fingerprint1[:,falses_cols_13])
names3 = names(fingerprint1[:,falses_cols_23])
all_unstable_colnames = unique(vcat(vcat(names1,names2),names3))

