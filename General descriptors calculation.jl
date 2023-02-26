# General descriptors calculation
using BSON
using Statistics
using DataFrames
using CSV
using PyCall
using Conda
using WAV
using Plots
y, fs = wavread(raw"C:\Windows\Media\Ring01.wav")

pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
alc = pyimport("rdkit.Chem.AllChem")

data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI-.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI+.csv", DataFrame)
data3 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI-.csv", DataFrame)
data4 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI+.csv", DataFrame)

data_M4_minus = (unique(data1,2))[!,[2,3,27,28,29,30,31]]
data_M4_plus = (unique(data2,2))[!,[2,3,27,28,29,30,31]]
data_M2_minus = data3[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_M2_plus = data4[!,[:name,:pH_aq,:logIE,:instrument,:source,:solvent,:doi]]
data_minus = vcat(data_M4_minus, data_M2_minus)
data_plus = vcat(data_M4_plus, data_M2_plus)


## Fingerprint calculation (function)##  (from name)
function padel_fp(rep)
    results = pcp.get_compounds(rep[1,1], "name")[1]
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    joined = hcat(DataFrame(rep[1,:]), desc_p)
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            results = pcp.get_compounds(rep[i,1], "name")[1]
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            joined_p = hcat(DataFrame(rep[i,:]), desc_p_temp)
            joined = append!(joined,joined_p)
            println(i)
        catch
            continue
        end
    end
    return joined
end

## Fingerprint calculation (function)## (from name with sids)
function padel_fromname(rep)
    sids = pcp.get_sids(rep[1,1], "name")
    results = pcp.Compound.from_cid(sids[1]["CID"])
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    inchi = DataFrame(INCHIKEY=[results.inchikey])
    joined = hcat(hcat(DataFrame(rep[1,:]), inchi),desc_p)
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            sids = pcp.get_sids(rep[i,1], "name")
            results = pcp.Compound.from_cid(sids[1]["CID"])
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            inchi = DataFrame(INCHIKEY=[results.inchikey])
            joined_p = hcat(hcat(DataFrame(rep[i,:]), inchi), desc_p_temp)
            joined = append!(joined,joined_p)
            println(i)
        catch
            continue
        end
    end
    return joined
end

## Fingerprint calculation (function)## (from SMILES)
function padel_fromSMILES(rep,smilesvec)
    sids = pcp.get_sids(smilesvec[1], "smiles")
    results = pcp.Compound.from_cid(sids[1]["CID"])
    desc_p = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
    inchi = DataFrame(INCHIKEY=[results.inchikey])
    joined = hcat(hcat(DataFrame(rep[1,:]), inchi),desc_p)
    for i = 2:size(rep,1)
        if size(desc_p,1) >= i
            println("Error on compound $i by $(size(desc_p,1)-i)")
        end
        try
            sids = pcp.get_sids(smilesvec[i], "smiles")
            results = pcp.Compound.from_cid(sids[1]["CID"])
            desc_p_temp = DataFrame(pd.from_smiles(results.isomeric_smiles,fingerprints=true, descriptors=false))
            inchi = DataFrame(INCHIKEY=[results.inchikey])
            joined_p = hcat(hcat(DataFrame(rep[i,:]), inchi), desc_p_temp)
            joined = append!(joined,joined_p)
            println(i)
        catch
            continue
        end
    end
    return joined
end


## Fingerprint calculation (calc)##
fp_minus_12_name = padel_fromname(data_minus)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_w_inchikey2.csv",fp_minus_12_name)
#fp_plus_12_name = padel_fromname(data_plus)
#CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_fromnamecid_new.csv", fp_plus_12_name)

#fp_minus_12_1 = padel_fromname(data_M4_minus)
#fp_minus_12_2= padel_fromSMILES(data_M2_minus[:,:],data3[:,:SMILES])
#fp_minus_12_nameSMILES = vcat(fp_minus_12_1,fp_minus_12_2)
#CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnameSMILES_new.csv", fp_minus_12_nameSMILES)
fp_plus_12_1 = padel_fromname(data_M4_plus)
fp_plus_12_2= padel_fromSMILES(data_M2_plus[:,:],data4[:,:SMILES])
fp_plus_12_nameSMILES = vcat(fp_plus_12_1,fp_plus_12_2)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_w_inchikey.csv", fp_plus_12_nameSMILES)

## Comparison ##
fromnamecids = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromsids.csv", DataFrame)[:,1:7]
fromSMILES = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnameSMILES.csv", DataFrame)
big = data_minus
setdiff(eachrow(big),eachrow(small))
setdiff(eachrow(fromnamecids),eachrow(fromSMILES))

#minus comparison
minus_big = data_minus
minus_cid_old = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12.csv", DataFrame)
minus_cid_new = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnamecid_new.csv", DataFrame)
minus_cid_new_2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnamecid_new(2).csv", DataFrame)
minus_smiles_old = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnameSMILES.csv", DataFrame)
minus_smiles_new = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_fromnameSMILES_new.csv", DataFrame)

setdiff(minus_big[:,1],minus_best[:,1])

#plus comparison
plus_big = data_plus
plus_cid_old = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12.csv", DataFrame)
plus_cid_new = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_fromnamecid_new.csv", DataFrame)
plus_smiles_new = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_fromnameSMILES_new.csv", DataFrame)
plus_smiles_new_2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_fromnameSMILES_new(2).csv", DataFrame)

name_issues = setdiff(plus_big[:,1],plus_smiles_new[:,1])
println(name_issues[1:10])














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
