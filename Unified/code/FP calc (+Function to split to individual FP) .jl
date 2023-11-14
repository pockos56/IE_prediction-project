using CSV, DataFrames, PyCall, Conda, ProgressBars, Random, BSON, Statistics
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
alc = pyimport("rdkit.Chem.AllChem")

# Loading data
#data = CSV.read("/home/emcms/Data/Data Alex/IE_prediction/unified/data/filtered_data.csv", DataFrame)
data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\filtered_data.csv", DataFrame)
data_short = data[1:12,:]

# padelpy fingerprints
function from_smiles_vector(canonical_smiles::Vector)
    fp_padel = DataFrame(pd.from_smiles(canonical_smiles[1],fingerprints=true, descriptors=false))
    for i in ProgressBar(2:length(canonical_smiles))
        fp_padel = append!(fp_padel, DataFrame(pd.from_smiles(canonical_smiles[i],fingerprints=true, descriptors=false)))
    end
    return fp_padel
end
function from_smiles_vector2(rep::DataFrame)
    cids = pcp.get_compounds(rep[1,"INCHIKEY"], "inchikey")
    shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
    fp_padel = hcat(DataFrame(rep[1,:]), DataFrame(pd.from_smiles(shortest_cid.isomeric_smiles,fingerprints=true, descriptors=false)))
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

# Writing down the names of the FP types
fp_all = from_smiles_vector2(data_short)
fp_type1 = names(from_smiles_vector2(data_short))
fp_type2 = names(from_smiles_vector2(data_short))
fp_type3 = names(from_smiles_vector2(data_short))
fp_type4 = names(from_smiles_vector2(data_short))
fp_type5 = names(from_smiles_vector2(data_short))
fp_type6 = names(from_smiles_vector2(data_short))
fp_type7 = names(from_smiles_vector2(data_short))
fp_type8 = names(from_smiles_vector2(data_short))
fp_type9 = names(from_smiles_vector2(data_short))
fp_type10 = names(from_smiles_vector2(data_short))
fp_type11 = names(from_smiles_vector2(data_short))
fp_type12 = names(from_smiles_vector2(data_short))

# Saving them in one file
FP_types = [fp_type1[10:end]]
FP_types = hcat(FP_types,[fp_type2[10:end]])
FP_types = hcat(FP_types,[fp_type3[10:end]])
FP_types = hcat(FP_types,[fp_type4[10:end]])
FP_types = hcat(FP_types,[fp_type5[10:end]])
FP_types = hcat(FP_types,[fp_type6[10:end]])
FP_types = hcat(FP_types,[fp_type7[10:end]])
FP_types = hcat(FP_types,[fp_type8[10:end]])
FP_types = hcat(FP_types,[fp_type9[10:end]])
FP_types = hcat(FP_types,[fp_type10[10:end]])
FP_types = hcat(FP_types,[fp_type11[10:end]])
FP_types = hcat(FP_types,[fp_type12[10:end]])
BSON.@save("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP_types",FP_types)

# Filtering (load data)
all_FPs = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\data_w_all_FPs.csv", DataFrame)
validation_comps_harry = ["Acephate", "acephate","Monuron", "monuron","Monuron_161_p","Metalaxyl", "metalaxyl", "Metalaxyl_135_p", "Dicyclohexyl phthalate", "dicyclohexyl phthalate", "Lufenuron", "Tributylamine", "tributylamine", "Debrisoquin sulfate"]
validation_comps_helen = ["Amitrole", "Aspartame", "Atrazine", "Caffeine","Caffeine ", "caffeine", "Chlorothiazide", "Dimethyl phthalate", "dimethyl phthalate", "Efavirenz","Emamectin B1a", "Guanylurea", "Histamine", "L-alanine", "Alanine", "alanine", "L-phenylalanine", "phenylalanine","Phenylalanine", "atrazine", "Methamidophos","Methamidophos ", "methamidophos", "Metolachlor","Metolachlor_268_p", "Progesterone","Progesterone_3255_p", "progesterone", "Rifaximin", "Spinosad A", "TCMTB","Benthiazole"]
validation_comps = sort(unique(vcat(validation_comps_harry, validation_comps_helen)))

# Filtering (erasing)
vec = []
for i in ProgressBar(1:length(validation_comps))
    vec = append!(vec, findall(x->x .== validation_comps[i], all_FPs[:,:name]))
end
filtered_FPs = deepcopy(all_FPs)
deleteat!(filtered_FPs,sort(unique(vec)))                                                          # Erase

# Filtering (Checking if done properly)
vec = []
for i in (1:length(validation_comps))
    vec = append!(vec, findall(x->x .== validation_comps[i], filtered_FPs[:,:name]))        # Checking if the filter has erased all validation set compounds
end
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\filtered_FPs.csv", filtered_FPs)

# Splitting the dataframe in 12 DataFrames
filtered_FPs = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\filtered_FPs.csv", DataFrame)
x1 = unique(filtered_FPs[:,:name])
function split_fp(fp_all::DataFrame; info_first::Int = 1, info_last::Int = 9, save_CSV::Bool=false, path::String="")
    BSON.@load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\FP_types",FP_types)
    # Info
    info = fp_all[:,info_first:info_last]
    # Fingerprints
    fp1 = hcat(info, fp_all[:,FP_types[1]])
    fp2 = hcat(info, fp_all[:,FP_types[2]])
    fp3 = hcat(info, fp_all[:,FP_types[3]])
    fp4 = hcat(info, fp_all[:,FP_types[4]])
    fp5 = hcat(info, fp_all[:,FP_types[5]])
    fp6 = hcat(info, fp_all[:,FP_types[6]])
    fp7 = hcat(info, fp_all[:,FP_types[7]])
    fp8 = hcat(info, fp_all[:,FP_types[8]])
    fp9 = hcat(info, fp_all[:,FP_types[9]])
    fp10 = hcat(info, fp_all[:,FP_types[10]])
    fp11 = hcat(info, fp_all[:,FP_types[11]])
    fp12 = hcat(info, fp_all[:,FP_types[12]])
    if save_CSV == true
        if path == ""
            error("Please enter a valid path to save the CSV files")
        end
        for i in ProgressBar(1:12)
            fp_temp = hcat(info, fp_all[:,FP_types[i]])
            path_to_save = path * "\\FP$i.CSV"
            CSV.write(path_to_save, fp_temp)
        end
    end
    return fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12
end

fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12 = split_fp(filtered_FPs, save_CSV=true, path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\unified\\data\\Fingerprints")

# General descriptors calculation

## Denice compression FPs
function convertPubChemFPs(ACfp::DataFrame, PCfp::DataFrame)
    FP1tr =  convert.(Float64,ACfp)
    pubinfo = convert.(Int,Matrix(PCfp))

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
        println("Complete")

        return custom_FPs_df
end
PC_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP6.csv", DataFrame)
AC_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP12.csv", DataFrame)
compressed_FP = hcat(AC_FP[:,1:9], convertPubChemFPs(AC_FP[:,10:end], PC_FP[:,10:end]))
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP14.csv", compressed_FP)

## Morgan FP ## 
function morgan(rep)
    #results = pcp.get_compounds(rep[1,1], "name")[1]
    m1 = alc.MolFromSmiles(rep[1,:SMILES_cor])
    fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
    fp_vector = Int.((ones(1024).*1821)')
    for i = 1:1024
        fp_vector[i] = fp[i]
    end
    for j in ProgressBar(2:size(rep,1))
        m1 = alc.MolFromSmiles(rep[j,:SMILES_cor])
        fp = alc.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
        fp_vector_temp = Int.((ones(1024).*1821)')
        for i = 1:1024
            fp_vector_temp[i] = fp[i]
        end
        fp_vector = vcat(fp_vector,fp_vector_temp)
    end    
    return fp_vector
end

AC_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP12.csv", DataFrame)
morgan_FP = morgan(AC_FP[:,1:9])
morgan_whole =hcat(AC_FP[:,1:9], DataFrame(morgan_FP,:auto))
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\FP15.csv", morgan_whole)
