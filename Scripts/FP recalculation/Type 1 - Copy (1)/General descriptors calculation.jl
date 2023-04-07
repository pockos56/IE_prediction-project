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
pwd()

## 5-Apr-2023 ##
function padel_fp(ESI::Int, FP::String ; allowsave=true)
    if ESI==-1
        ESI_name = "neg"
    elseif ESI==+1
        ESI_name = "pos"
    else error("ESI should be +1 or -1")
    end
    
    rep = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)[:,1:8]
    results = pcp.get_compounds(rep[1,:INCHIKEY], "inchikey")[1]
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    joined = hcat(DataFrame(rep[1,:]), desc_p)
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            results = pcp.get_compounds(rep[i,:INCHIKEY], "inchikey")[1]
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            joined_p = hcat(DataFrame(rep[i,:]), desc_p_temp)
            joined = append!(joined,joined_p)
            println("$i/$(size(rep,1))")
        catch
            continue
        end
    end
    return joined
    if allowsave==true
        CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_$(ESI_name)_$FP.csv", joined)
    end
end

padel_neg = padel_fp(-1,"01", allowsave=true)
padel_pos = padel_fp(+1,"01", allowsave=true)










## Morgan FP ##    
function morgan(rep)
    results = pcp.get_compounds(rep[1,1], "name")[1]
    m1 = alc.MolFromSmiles(results.isomeric_smiles)
    fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
    fp_vector = (ones(1024).*1821)'
    for i = 1:1024
        fp_vector[i] = fp[i]
    end
    for j = 2:size(rep,1)
        results = pcp.get_compounds(rep[j,1], "name")[1]
        m1 = alc.MolFromSmiles(results.isomeric_smiles)
        fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
        fp_vector_temp = (ones(1024).*1821)'
        for i = 1:1024
            fp_vector_temp[i] = fp[i]
        end
        fp_vector = vcat(fp_vector,fp_vector_temp)
        println("End of $j compound")
    end    
return fp_vector
end
morgan_minus = morgan(data_minus)
morgan_plus = morgan(data_plus)

morgan_minus_ = DataFrame(hcat(data_minus[:,1], morgan_minus), :auto)
morgan_plus_ = DataFrame(hcat(data_plus[:,1], morgan_plus), :auto)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\morgan_minus.csv", morgan_minus_)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\morgan_plus.csv", morgan_plus_)
