using CSV
using DataFrames
using StatsBase


CNL_ref = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Pred pred validation\\CNL_ref.csv", DataFrame)
CNL_ref_susconv = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Pred pred validation\\CNL_ref_susconv.csv", DataFrame)

names(CNL_ref)
names(CNL_ref_susconv)

unique(CNL_ref[:,:INCHIKEY])
unique(CNL_ref_susconv[:,:INCHIKEY])

unique(CNL_ref[:,:SMILES])
unique(CNL_ref_susconv[:,:SMILES])

CNL_ref_susconv[CNL_ref_susconv[:,:INCHIKEY] .!= "PRLVTUNWOQKEAI-UHFFFAOYSA-N",:]   #[:,4:end]
# Filtering
CNL_ref_susconv_temp =  CNL_ref_susconv[CNL_ref_susconv[:,:INCHIKEY] .!= "",:] 
CNL_ref_susconv_temp = CNL_ref_susconv_temp[CNL_ref_susconv_temp[:,:INCHIKEY] .!= "NA",:]   
CNL_ref_susconv_temp = CNL_ref_susconv_temp[CNL_ref_susconv_temp[:,:INCHIKEY] .!= "N/A",:]   
CNL_ref_susconv_temp = CNL_ref_susconv_temp[CNL_ref_susconv_temp[:,:INCHIKEY] .!= "104.05",:]   
CNL_ref_susconv_temp = CNL_ref_susconv_temp[ismissing.(CNL_ref_susconv_temp[:,:INCHIKEY]) .== false,:]
#

unique(CNL_ref_susconv_temp[:,:MZ_VALUES])
map = sort(countmap(CNL_ref_susconv_temp[:,:MZ_VALUES]), rev=false)
duplicate_frags = convert.(String, keys(map))[values(map) .== 2]
CNL_ref_susconv_temp[CNL_ref_susconv_temp[:,:MZ_VALUES] .== duplicate_frags[13],:][:,6:end] 
unique(CNL_ref_susconv_temp[:,"MZ_INT"])

using Pkg
Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")

using IE_prediction
logIE_from_CNLs([20.3],200.5,"positive",10)
