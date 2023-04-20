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
padel_neg = padel_fp(-1,"12", allowsave=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_neg_12.csv", padel_neg)
padel_pos = padel_fp(+1,"12", allowsave=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_pos_12.csv", padel_pos)

## Denice compression FPs
function AC_and_compressedPC(ESI)
    if ESI==-1
        ESI_name = "neg"
    elseif ESI==+1
        ESI_name = "pos"
    else error("ESI should be +1 or -1")
    end
    function getFPsAndOthers(raw)
        # set[ismissing.(set[:,"XLogP"]),"XLogP"] .= NaN
        # set[ismissing.(set[:,"MLogP"]),"MLogP"] .= NaN
        # set[ismissing.(set[:,"ALogP"]),"ALogP"] .= NaN
        # set[ismissing.(set[:,"ALogp2"]),"ALogp2"] .= NaN
        # set[ismissing.(set[:,"CrippenLogP"]),"CrippenLogP"] .= NaN
        set = hcat(raw, DataFrame(SMILES = missings(size(raw,1))))
        set[!,:SMILES]=convert(Vector{Union{String,Missing}}, set[!,:SMILES])

        pubchemFPs = "PubchemFP" .* string.(collect(115:262))
        indStart = size(set,2)+1
        indFPs = []
        if indStart > 20
            println("error all variables provided")
            return set
        end
        for i = 1:size(set,1)
            if ismissing(set[i,"SMILES"])
                #get smiles from Inchikey
                comp = pcp.get_compounds(set[i,"INCHIKEY"], "inchikey")
                if length(comp) > 1
                    len = ones(length(comp)).*Inf
                    for s = 1:length(comp)
                        len[s] = length(comp[s].canonical_smiles)
                    end
                    set[i, "SMILES"] = comp[argmin(len)].canonical_smiles
                elseif  isempty(comp)
                    continue
                else
                    set[i, "SMILES"] = comp[1].canonical_smiles
                end
            end
            desc_p = []
            try
                desc_p = DataFrame(pd.from_smiles(set[i,"SMILES"],fingerprints=true, descriptors = false))
            catch
                continue
            end
            if isempty(desc_p)
                if i==1
                    println("Error at first iteration")
                    return
                end
                continue
            end
            #get FPs
            if i == 1
                #expand dataframe
                for f = 1:size(desc_p,2)
                    if contains(names(desc_p)[f],"APC2") || any(names(desc_p)[f] .== pubchemFPs)
                        indFPs = [indFPs;f]
                        set[!,names(desc_p)[f]] = fill("",size(set,1))
                    end
                end
                set[i,indStart:end] = desc_p[1,indFPs]
            else
                set[i,indStart:end] = desc_p[1,indFPs]
            end
            println("$i/$(size(set,1)) for ESI $(ESI)")
        end
        return set
    end
    function convertPubChemFPs(ACfp::DataFrame, PCfp::DataFrame)
        FP1tr =  parse.(Float64,ACfp)
        pubinfo = parse.(Int,Matrix(PCfp))

        #ring counts
        FP1tr[!,"PCFP-r3"] = pubinfo[:,1]
        FP1tr[!,"PCFP-r3"][pubinfo[:,8] .== 1] .= 2
        FP1tr[!,"PCFP-r4"] = pubinfo[:,15]
        FP1tr[!,"PCFP-r4"][pubinfo[:,22] .== 1] .= 2
        FP1tr[!,"PCFP-r5"] = pubinfo[:,29]
        FP1tr[!,"PCFP-r5"][pubinfo[:,36] .== 1] .= 2
        FP1tr[!,"PCFP-r5"][pubinfo[:,43] .== 1] .= 3
        FP1tr[!,"PCFP-r5"][pubinfo[:,50] .== 1] .= 4
        FP1tr[!,"PCFP-r5"][pubinfo[:,57] .== 1] .= 5

        FP1tr[!,"PCFP-r6"] = pubinfo[:,64]
        FP1tr[!,"PCFP-r6"][pubinfo[:,71] .== 1] .= 2
        FP1tr[!,"PCFP-r6"][pubinfo[:,78] .== 1] .= 3
        FP1tr[!,"PCFP-r6"][pubinfo[:,85] .== 1] .= 4
        FP1tr[!,"PCFP-r6"][pubinfo[:,92] .== 1] .= 5
        FP1tr[!,"PCFP-r7"] = pubinfo[:,99]
        FP1tr[!,"PCFP-r7"][pubinfo[:,106] .== 1] .= 2
        FP1tr[!,"PCFP-r8"] = pubinfo[:,113]
        FP1tr[!,"PCFP-r8"][pubinfo[:,120] .== 1] .= 2
        FP1tr[!,"PCFP-r9"] = pubinfo[:,127]
        FP1tr[!,"PCFP-r10"] = pubinfo[:,134]

        #minimum number of type of rings
        arom = zeros(size(pubinfo,1))
        arom[(arom .== 0) .& (pubinfo[:,147] .== 1)] .= 4
        arom[(arom .== 0) .& (pubinfo[:,145] .== 1)] .= 3
        arom[(arom .== 0) .& (pubinfo[:,143] .== 1)] .= 2
        arom[(arom .== 0) .& (pubinfo[:,141] .== 1)] .= 1
        FP1tr[!,"minAromCount"] = arom
        het = zeros(size(pubinfo,1))
        het[(het .== 0) .& (pubinfo[:,148] .== 1)] .= 4
        het[(het .== 0) .& (pubinfo[:,146] .== 1)] .= 3
        het[(het .== 0) .& (pubinfo[:,144] .== 1)] .= 2
        het[(het .== 0) .& (pubinfo[:,142] .== 1)] .= 1
        FP1tr[!,"minHetrCount"] = het

        custom_FPs_df = DataFrame(convert.(Int16, Matrix(FP1tr)), names(FP1tr))
        println("ESI $(ESI) Complete")

        return custom_FPs_df
    end

    raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)[:,1:8]
    X = getFPsAndOthers(raw)
    fingerprints_df = convertPubChemFPs(X[:,10:10+779], X[:,10+780:end])
    final_df = hcat(X[:,1:9],fingerprints_df)
    return final_df
end

custom_neg = AC_and_compressedPC(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_neg_15.csv", custom_neg)
custom_pos = AC_and_compressedPC(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Scripts\\FP recalculation\\Results\\padel_M2M4_pos_15.csv", custom_pos)


