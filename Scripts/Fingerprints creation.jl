using CSV
using DataFrames
using BSON
using PyCall
using Conda
pd = pyimport("padelpy")


Norm_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman data\\Norm_descriptors_part1.csv", DataFrame)
for i = 2:20
    Norm_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman data\\Norm_descriptors_part$i.csv", DataFrame)
    Norm_raw = vcat(Norm_raw, Norm_temp)
    println("$i/20")
end

## Fingerprint calculation (function)##
function padel_desc(rep)
    desc_p = DataFrame(pd.from_smiles(rep[1,2],fingerprints=true, descriptors=false))
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            desc_p_temp = DataFrame(pd.from_smiles(rep[i,2],fingerprints=true, descriptors=false))
            desc_p = vcat(desc_p,desc_p_temp)
            println(i)
        catch
            continue
        end
    end
    desc_full = hcat(rep[:,1:5],desc_p)
    return desc_full
end

## Fingerprint calculation (calculation) ##
Norman_FP = padel_desc(Norm_raw[1:5,:])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\RTI_prediction\\Norman_FP.csv", Norman_FP)
