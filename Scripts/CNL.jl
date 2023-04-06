## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using PyCall
using Conda
using JLD
using DataStructures

## load files ##
M2M4_neg = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_neg_12_w_inchikey.csv", DataFrame)
M2M4_pos = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_pos_12_w_inchikey.csv", DataFrame)
amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL datasets\\Amide_CNLs.csv", DataFrame)
best_CNLs = sort(JLD.load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLs.jld")["CNLmax"])

# Preprocessing
# Changing the MassBank dataset to the Amide format
function create_CNLIE_dataset_MB(ESI)
    function MB_df(input)
        function cnl_to_fingerprint(str::String, precursor::Float64; threshold::Float64=0.01)
            # Convert string to vector{float64}
            str = replace(str, r"Any(\[.*?\])" => s"\1")
            val_vec = eval(Meta.parse(str))
            # Define the range and step size for the fingerprint
                #fingerprint_range = 0:0.01:1000
            fingerprint_range = best_CNLs
                
            # Initialize the fingerprint dictionary
            fingerprint_dict = sort(Dict([(x, 0) for x in fingerprint_range]))
            # Loop over the values and update the fingerprint dictionary
            for v in val_vec
                # Find the nearest index in the fingerprint_range (FIX TO SET TO ONE WITH A BINNING OF 0.01)
                if findmin(abs.(fingerprint_range .- v))[1] <= threshold
                    idx = findmin(abs.(fingerprint_range .- v))[2]
                    # Increment the count for that index in the fingerprint dictionary
                    fingerprint_dict[fingerprint_range[idx]] = 1
                end
            end
            # Convert the fingerprint dictionary to a DataFrame
            dict_str = OrderedDict(string(k) => v for (k, v) in fingerprint_dict)
            fingerprint_df = DataFrame(dict_str)
            # Change all values higher than the precursor ion to -1
            idx_precursor = findmin(abs.(fingerprint_range .- precursor))[2]
            fingerprint_df[:,idx_precursor+1:end] .= -1
            return fingerprint_df
        end
        # str = "1.01, 12.3, 15.5, 114.07"
        
        df = DataFrame()
        for i = 1:size(input,1)
             df_temp_CNL = vector_to_fingerprint(input[i,:neutral_loss], input[i,:precursor_ion])
             df_temp_info = DataFrame((hcat(hcat(hcat(hcat(i,input[i,:name]),hcat(0,input[i,:INCHIKEY])),hcat(input[i,:SMILES],input[i,:formula])),input[i,:exact_mass])),names(amide_raw[2,1:7]))
             df_temp = hcat(df_temp_info, df_temp_CNL)
             append!(df, df_temp)
            #println("Convert to DF-Compound $i out of $(size(input,1))")
        end
        return df
    end

    if ESI == -1
        ESI_name = "NEG"
        IE_raw = M2M4_neg
    elseif ESI == 1
        ESI_name = "POS"
        IE_raw = M2M4_pos
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MassBankSpec_$ESI_name.csv", DataFrame)
    chunk_size = 15
    dataset = DataFrame()
    for chunk = 1:chunk_size
        range = Int64.(round.(LinRange(1,size(MB,1),(chunk_size+1))))
        MB_temp = MB_df(MB[range[chunk]:range[chunk+1],:])

        MB_temp[!,:FORMULA] = convert.(String, MB_temp[!,:FORMULA])
        select!(MB_temp, Not([:RI_pred]))
        insertcols!(MB_temp, 4, :pH_aq => -666.6)  # That's the pH column
        insertcols!(MB_temp, 4, :logIE => -666.6)  # That's the IE column
        #info_end = findfirst(x->x .== "1.01", names(MB_temp))-1
        #indices = vcat(collect(1:info_end), findall(x -> x in Float32.(best_CNLs), Float32.(collect(0:0.01:1000))) .+ info_end)

        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = MB_temp[MB_temp[:,:INCHIKEY] .== key, :]# That's the CNLs of compound i
            CNL_set_temp[:,:pH_aq] .= IE_raw[i,:pH_aq]        # That's the pH of compound i
            CNL_set_temp[:,:logIE] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            println("MB part $chunk/$(chunk_size) : Compound $i/$(size(IE_raw,1))")
        end
    end
    return dataset
end
function create_CNLIE_dataset_amide(ESI)
    if ESI == -1
        ESI_name = "neg"
        IE_raw = M2M4_neg
    elseif ESI == 1
        ESI_name = "pos"
        IE_raw = M2M4_pos
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    amide_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Amide_CNLs.csv", DataFrame)[:,:1:100008]
    chunk_size = 5
    dataset = DataFrame()
    for chunk = 1:chunk_size
        range = Int64.(round.(LinRange(1,size(amide_raw,1),(chunk_size+1))))
        amide_temp = amide_raw[range[chunk]:range[chunk+1],:]
        amide_temp[!,:FORMULA] = convert.(String, amide_temp[!,:FORMULA])
        select!(amide_temp, Not([:RI_pred]))
        insertcols!(amide_temp, 4, :pH_aq => -666.6)  # That's the pH column
        insertcols!(amide_temp, 4, :logIE => -666.6)  # That's the IE column
        info_end = findfirst(x->x .== "0", names(amide_temp))-1
        indices = vcat(collect(1:info_end), findall(x -> x in Float32.(best_CNLs), Float32.(collect(0:0.01:1000))) .+ info_end)

        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = amide_temp[amide_temp[:,:INCHIKEY] .== key, indices] # That's the CNLs of compound i
            CNL_set_temp[:,:pH_aq] .= IE_raw[i,:pH_aq]        # That's the pH of compound i
            CNL_set_temp[:,:logIE] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            println("Amide $ESI_name -  $chunk/$(chunk_size) : Compound $i/$(size(IE_raw,1))")
        end
    end
    return dataset
end
function create_CNLIE_dataset_norman(ESI)
    if ESI == -1
        ESI_name = "neg"
        IE_raw = M2M4_neg
    elseif ESI == 1
        ESI_name = "pos"
        IE_raw = M2M4_pos
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    dataset = DataFrame()
    for filename = 1:12
        norman_temp = (CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Norman_CNLs.$filename", DataFrame))[:,1:100008]  #Load norman
        norman_temp[!,:FORMULA] = convert.(String,norman_temp[!,:FORMULA])
        select!(norman_temp, Not([:leverage, :RI_pred]))
        insertcols!(norman_temp, 4, :pH_aq => -666.6)  # That's the pH column
        insertcols!(norman_temp, 4, :logIE => -666.6)  # That's the IE column
        info_end = findfirst(x->x .== "0", names(norman_temp))-1
        indices = vcat(collect(1:info_end), findall(x -> x in Float32.(best_CNLs), Float32.(collect(0:0.01:1000))) .+ info_end)

        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = norman_temp[norman_temp[:,:INCHIKEY] .== key, indices] # That's the CNLs of compound i
            CNL_set_temp[:,:pH_aq] .= IE_raw[i,:pH_aq]        # That's the IE of compound i (REMOVE)
            CNL_set_temp[:,:logIE] .= IE_raw[i,:logIE]        # That's the IE of compound i (REMOVE)
            append!(dataset, CNL_set_temp)
            println("Norman $ESI_name - $filename/12 : Compound $i/$(size(IE_raw,1))")
        end
    end
    return dataset
end
function create_CNLIE_dataset_NIST(ESI)
    function NIST_df(input)
        function mz_to_fingerprint(str::String, precursor::Float64; threshold::Float64=0.01)
            # Convert string to vector{float64}
            str = replace(str, r"Any(\[.*?\])" => s"\1")
            mz_vec = eval(Meta.parse(str))
            CNL_vec = (precursor .- mz_vec)[(precursor .- mz_vec).>=0]
            # Define the range and step size for the fingerprint
            fingerprint_range = best_CNLs
                
            # Initialize the fingerprint dictionary
            fingerprint_dict = sort(Dict([(x, 0) for x in fingerprint_range]))
            # Loop over the values and update the fingerprint dictionary
            for v in CNL_vec
                # Find the nearest index in the fingerprint_range (FIX TO SET TO ONE WITH A BINNING OF 0.01)
                if findmin(abs.(fingerprint_range .- v))[1] <= threshold
                    idx = findmin(abs.(fingerprint_range .- v))[2]
                    # Increment the count for that index in the fingerprint dictionary
                    fingerprint_dict[fingerprint_range[idx]] = 1
                end
            end
            # Convert the fingerprint dictionary to a DataFrame
            dict_str = OrderedDict(string(k) => v for (k, v) in fingerprint_dict)
            fingerprint_df = DataFrame(dict_str)
            # Change all values higher than the precursor ion to -1
            idx_precursor = findmin(abs.(fingerprint_range .- precursor))[2]
            fingerprint_df[:,idx_precursor+1:end] .= -1
            return fingerprint_df
        end
        
        df = DataFrame()
        features_w_order = [:ACCESSION, :NAME,:COLLISION_ENERGY, :INCHIKEY, :SMILES, :FORMULA, :EXACT_MASS]
        for i = 1:size(input,1)
            df_temp_CNL = mz_to_fingerprint(input[i,:MZ_VALUES], input[i,:PRECURSOR_ION])
            df_temp_info = DataFrame(Matrix(DataFrame(input[i,features_w_order])),names(amide_raw[2,1:7]))
            df_temp = hcat(df_temp_info, df_temp_CNL)
            append!(df, df_temp)
            println("Convert to DF-Compound $i out of $(size(input,1))")
        end
        return df
    end

    NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)
    NIST_ESI = ()
    if ESI == -1
        ESI_name = "NEG"
        IE_raw = M2M4_neg
        NIST_ESI = vcat(NIST[Vector(NIST[:,:ION_MODE]).=="N",:],NIST[Vector(NIST[:,:ION_MODE]).=="NEGATIVE",:])
    elseif ESI == +1
        ESI_name = "POS"
        IE_raw = M2M4_pos
        NIST_ESI = vcat(NIST[Vector(NIST[:,:ION_MODE]).=="P",:],NIST[Vector(NIST[:,:ION_MODE]).=="POSITIVE",:])
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    
    chunk_size = 20
    dataset = DataFrame()
    range = Int64.(round.(LinRange(1,size(NIST_ESI,1),(chunk_size+1))))
    for chunk = 1:chunk_size
        NIST_temp = NIST_df(NIST_ESI[range[chunk]:range[chunk+1],:])

        NIST_temp[!,:FORMULA] = convert.(String, NIST_temp[!,:FORMULA])
        select!(NIST_temp, Not([:RI_pred]))
        insertcols!(NIST_temp, 4, :pH_aq => -666.6)  # That's the pH column
        insertcols!(NIST_temp, 4, :logIE => -666.6)  # That's the IE column
        #info_end = findfirst(x->x .== "1.01", names(NIST_temp))-1
        #indices = vcat(collect(1:info_end), findall(x -> x in Float32.(best_CNLs), Float32.(collect(0:0.01:1000))) .+ info_end)

        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,:INCHIKEY]           # That's the inchikey of compound i
            CNL_set_temp = NIST_temp[NIST_temp[:,:INCHIKEY] .== key, :]# That's the CNLs of compound i
            CNL_set_temp[:,:pH_aq] .= IE_raw[i,:pH_aq]        # That's the pH of compound i
            CNL_set_temp[:,:logIE] .= IE_raw[i,:logIE]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            println("NIST part $chunk/$(chunk_size) : Compound $i/$(size(IE_raw,1))")
        end
    end
    return dataset
end

# To run tonight 18-Mar-2023
MB_NEG = create_CNLIE_dataset_MB(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_NEG_selectedCNLs.csv", MB_NEG)
MB_POS = create_CNLIE_dataset_MB(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_POS_selectedCNLs.csv", MB_POS)
amide_NEG = create_CNLIE_dataset_amide(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_NEG_selectedCNLs.csv", amide_NEG)
amide_POS = create_CNLIE_dataset_amide(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_POS_selectedCNLs.csv", amide_POS)
norman_NEG = create_CNLIE_dataset_norman(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_NEG_selectedCNLs.csv", norman_NEG)
norman_POS = create_CNLIE_dataset_norman(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_POS_selectedCNLs.csv", norman_POS)
NIST_neg = create_CNLIE_dataset_NIST(-1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_NEG_selectedCNLs.csv", NIST_neg)
NIST_pos = create_CNLIE_dataset_NIST(+1)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_POS_selectedCNLs.csv", NIST_pos)

# We need to keep every entry as is, without combining.
# Talk with Vika to introduce the most significant CNLs
# Add pH variable, add monoisotopic mass variable