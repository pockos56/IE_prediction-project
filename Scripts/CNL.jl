## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using PyCall
using Conda

## load files ##
M2M4_minus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_minus_12_w_inchikey.csv", DataFrame)
M2M4_plus = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Fingerprints\\padel_M2M4_plus_12_w_inchikey.csv", DataFrame)

# Preprocessing
# Changing the MassBank dataset to the Amide format
function MB_df(input)
    function vector_to_fingerprint(str::String,precursor::Float64)
        # Convert string to vector{float64}
        str = replace(str, r"Any(\[.*?\])" => s"\1")
        val_vec = eval(Meta.parse(str))
        # Define the range and step size for the fingerprint
        fingerprint_range = 0:0.01:1000
        # Initialize the fingerprint dictionary
        fingerprint_dict = Dict([(x, 0) for x in fingerprint_range])
        # Loop over the values and update the fingerprint dictionary
        for v in val_vec
            # Find the nearest index in the fingerprint_range
            idx = findmin(abs.(fingerprint_range .- v))[2]
            # Increment the count for that index in the fingerprint dictionary
            fingerprint_dict[fingerprint_range[idx]] = 1
        end
        # Convert the fingerprint dictionary to a DataFrame
        dict_str = Dict(string(k) => v for (k, v) in fingerprint_dict)
        fingerprint_df = DataFrame(dict_str)
        # Change all values higher than the precursor ion to -1
        idx_precursor = findmin(abs.(fingerprint_range .- precursor))[2]
        fingerprint_df[:,idx_precursor+1:end] .= -1
        return fingerprint_df
    end
    
    df = DataFrame()
    for i = 1:size(input,1)
         df_temp_CNL = DataFrame(Matrix(vector_to_fingerprint(input[i,:neutral_loss], input[i,:precursor_ion])),names(amide_raw[2,8:100008]))
         info_temp = (hcat(hcat(hcat(hcat(i,input[i,:name]),hcat(0,input[i,:INCHIKEY])),hcat(input[i,:SMILES],input[i,:formula])),input[i,:exact_mass]))
         df_temp_info = DataFrame(info_temp,names(amide_raw[2,1:7]))
         df_temp = hcat(df_temp_info, df_temp_CNL)
         append!(df, df_temp)
        println("Compound $i out of $(size(input,1))")
    end
    return df
end
function create_CNLIE_dataset_MB(ESI)
    if ESI == -1
        ESI_name = "NEG"
        IE_raw = M2M4_minus
    elseif ESI == 1
        ESI_name = "POS"
        IE_raw = M2M4_plus
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MassBankSpec_$ESI_name.csv", DataFrame)
    dataset_whole = DataFrame()
    chunk_size = 15
    for chunk = 1:chunk_size
        range = Int64.(round.(LinRange(1,size(MB,1),(chunk_size+1))))
        MB_temp = MB_df(MB[range[chunk]:range[chunk+1],:])
        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        dataset = DataFrame()
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = MB_temp[MB_temp[:,:INCHIKEY] .== key, :]# That's the CNLs of compound i
            CNL_set_temp[:,:RI_pred] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            println("MB part $chunk/$(chunk_size) : Compound $i/$(size(IE_raw,1))")
        end
        rename!(dataset, :RI_pred => :logIE)
        append!(dataset_whole,dataset)
    end
    return dataset_whole
end
function create_CNLIE_dataset_amide(ESI)
    if ESI == -1
        ESI_name = "minus"
        IE_raw = M2M4_minus
    elseif ESI == 1
        ESI_name = "plus"
        IE_raw = M2M4_plus
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Amide_CNLs.csv", DataFrame)
    dataset_whole = DataFrame()
    chunk_size = 5
    for chunk = 1:chunk_size
        range = Int64.(round.(LinRange(1,size(amide_raw,1),(chunk_size+1))))
        amide_temp = amide_raw[range[chunk]:range[chunk+1],:]
        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = amide_temp[amide_temp[:,:INCHIKEY] .== key, 1:100008] # That's the CNLs of compound i
            CNL_set_temp[:,:RI_pred] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset_whole, CNL_set_temp)
            println("Amide part $chunk/$(chunk_size) : Compound $i/$(size(IE_raw,1))")
        end
        rename!(dataset, :RI_pred => :logIE)
        append!(dataset_whole,dataset)
    end
    return dataset_whole
end
function create_CNLIE_dataset_norman(ESI)
    if ESI == -1
        ESI_name = "minus"
        IE_raw = M2M4_minus
    elseif ESI == 1
        ESI_name = "plus"
        IE_raw = M2M4_plus
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    dataset_whole = DataFrame()
    for filename = 1:12
        norman_temp = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs.$filename", DataFrame)
        select!(norman_temp, Not([:leverage, :Pred_RTI_Pos_ESI, :Pred_RTI_Neg_ESI]))
        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        dataset = DataFrame()
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = norman_temp[norman_temp[:,:INCHIKEY] .== key, 1:100008] # That's the CNLs of compound i
            CNL_set_temp[:,:RI_pred] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            println("Norman part $filename/12 : Compound $i/$(size(IE_raw,1))")
        end
        rename!(dataset, :RI_pred => :logIE)
        append!(dataset_whole,dataset)
    end
    return dataset_whole
end

# To run tonight 18-Mar-2023
MB_POS = create_CNLIE_dataset_MB(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_MB_POS.csv", MB_POS)
MB_NEG = create_CNLIE_dataset_MB(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_MB_NEG.csv", MB_NEG)
amide_POS = create_CNLIE_dataset_amide(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_amide_POS.csv", amide_POS)
amide_NEG = create_CNLIE_dataset_amide(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_amide_NEG.csv", amide_NEG)
norman_POS = create_CNLIE_dataset_norman(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_norman_POS.csv", norman_POS)
norman_NEG = create_CNLIE_dataset_norman(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLIE_norman_NEG.csv", norman_NEG)

# We need to keep every entry as is, without combining.