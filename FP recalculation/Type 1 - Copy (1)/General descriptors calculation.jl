# General descriptors calculation
using BSON
using Statistics
using DataFrames
using CSV
using PyCall
using Conda
using Plots
using DataStructures
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
alc = pyimport("rdkit.Chem.AllChem")
pwd()

## 5-Apr-2023 ##
function padel_fp(ESI::Int, FP::String)
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
end
padel_neg = padel_fp(-1,"13")
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_neg_13.csv", padel_neg)
padel_pos = padel_fp(+1,"13")
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_pos_13.csv", padel_pos)



## Morgan FP BitVector##    
function morgan(ESI; num_bits::Int=1024)
    if ESI==-1
        ESI_name = "neg"
    elseif ESI==+1
        ESI_name = "pos"
    else error("ESI should be +1 or -1")
    end
    
    rep = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)[:,1:8]

    results = pcp.get_compounds(rep[1,:INCHIKEY], "inchikey")[1]
    m1 = alc.MolFromSmiles(results.isomeric_smiles)
    fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=num_bits)
    fp_vector = Int.(ones(num_bits).*3)'
    for i = 1:num_bits
        fp_vector[i] = fp[i]
    end
    full_vector = hcat(DataFrame(rep[1,:]), DataFrame(fp_vector,:auto))
    for j = 2:size(rep,1)
        results = pcp.get_compounds(rep[j,:INCHIKEY], "inchikey")[1]
        m1 = alc.MolFromSmiles(results.isomeric_smiles)
        fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=num_bits)
        fp_vector_temp = Int.(ones(num_bits).*3)'
        for i = 1:num_bits
            fp_vector_temp[i] = fp[i]
        end
        full_vector_temp = hcat(DataFrame(rep[j,:]), DataFrame(fp_vector_temp,:auto))
        full_vector = append!(full_vector, full_vector_temp)
        println("End of $j/$(size(rep,1)) compound")
    end    
return full_vector
end
morgan_neg = morgan(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_neg_14.csv", morgan_neg)
morgan_pos = morgan(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_pos_14.csv", morgan_pos)
