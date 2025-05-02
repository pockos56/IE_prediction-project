using CSV, DataFrames, ProgressBars, DataStructures, FreqTables, Statistics

# Create a shorter (filtered) Internal database with only the relevant compounds
data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", DataFrame)
Internal_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)
Internal_filtered = Internal_raw[findall(x -> x in unique(data[:,"INCHIKEY"]), Internal_raw[:,:INCHIKEY]),:]
common_INCHIKEYS_indx = findall(x -> x in unique(data[:,"INCHIKEY"]), Internal_filtered[:,:INCHIKEY])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", Internal_filtered)
Internal_filtered = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", DataFrame)

# Load data
data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", DataFrame)
Internal_filtered = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", DataFrame)
best_CNLs_neg = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\CNLmax_Hadducts_neg.CSV", DataFrame)[1:500,1]
best_CNLs_pos = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\CNLmax_Hadducts_pos.CSV", DataFrame)[1:500,1]
#mode = "mean"

# 1-Dec-2023 Binning and consensus spectra for MMM 
function create_CNLIE_dataset_NIST_Unified_mode(mode::String ;ESI=+1, threshold=0.02, HiValSetToZero::Bool=true, chunk_size=10)
    function NIST_df(input)
        function mz_to_fingerprint(str::String, precursor::Float64, ESI::Int; threshold)
            # Convert string to vector{float64}
            str = replace(str, r"Any(\[.*?\])" => s"\1")
            mz_vec = eval(Meta.parse(str))
            CNL_vec = round.((precursor .- mz_vec)[(precursor .- mz_vec).>=0],digits = 4)
            # Define the range and step size for the fingerprint
            fingerprint_range_raw = []
            if ESI == -1
                fingerprint_range_raw = round.(best_CNLs_neg, digits=2)
            elseif ESI == +1
                fingerprint_range_raw = round.(best_CNLs_pos, digits=2)
            else
                error("Put ESI to +1/-1")
            end
            distance_threshold = threshold - 0.01
            fingerprint_range = fingerprint_range_raw[1:2]
            distance=[]
            #fingerprint_range = vcat(fingerprint_range, 18.0)
            for i = 3:length(fingerprint_range_raw)
                distance = round.(abs.(fingerprint_range_raw[i] .- fingerprint_range),digits=3)
                if all(distance .> distance_threshold) #isempty(distance[distance .<= distance_threshold])
                    push!(fingerprint_range, fingerprint_range_raw[i])
                end
            end
                
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
            idx_precursor = findmin(abs.(Meta.parse.(names(fingerprint_df)) .- precursor))[2]
            if HiValSetToZero == false
                fingerprint_df[:,idx_precursor+1:end] .= -1         # Set to -1, but could be changed
            elseif HiValSetToZero == true
                fingerprint_df[:,idx_precursor+1:end] .= 0         # Set to zero, but could be changed
            end
            return fingerprint_df
        end
        
        df = DataFrame()
        features_w_order = [:ACCESSION, :NAME,:COLLISION_ENERGY, :INCHIKEY, :SMILES, :FORMULA, :EXACT_MASS]
        for i = 1:size(input,1)
            df_temp_CNL = mz_to_fingerprint(input[i,:MZ_VALUES], input[i,:PRECURSOR_ION], ESI; threshold)

            df_temp_info = DataFrame(Matrix(DataFrame(input[i,features_w_order])),["Column1","NAME","RI_pred","INCHIKEY","SMILES","FORMULA","MONOISOMASS"])
            df_temp = hcat(df_temp_info, df_temp_CNL)
            append!(df, df_temp)
            # histogram(Vector(df[1,8:end]))
            if (i % 500)==0
            println("NIST $mode Convert to DF-Compound $i out of $(size(input,1))")
            end
        end
        return df
    end

    data_mode = DataFrame()
    data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", DataFrame)
    insertcols!(data, 5, "rounded_pH" => round.(data[:,"pH.aq."]))
    for unique_inchikey in unique(data[:,"INCHIKEY"])
    data_comp = data[findall(x->x .== unique_inchikey, data[:,"INCHIKEY"]),:]
        for unique_pH in unique(data_comp[:,"rounded_pH"])
            data_comp_pH = data_comp[findall(x->x .== unique_pH, data_comp[:,"rounded_pH"]),:]
            data_mode_temp = deepcopy(DataFrame(data_comp_pH[1,:]))
            data_mode_temp[1,"pH.aq."] = mean(data_comp_pH[:,"pH.aq."])
            if mode == "min"
                data_mode_temp[1,"unified_IEs"] = minimum(data_comp_pH[:,"unified_IEs"])
            elseif mode == "max"
                data_mode_temp[1,"unified_IEs"] = maximum(data_comp_pH[:,"unified_IEs"])
            elseif mode == "mean"
                data_mode_temp[1,"unified_IEs"] = mean(data_comp_pH[:,"unified_IEs"])
            else error("Set mode argument to mean, max, or min")
            end  
            data_mode = append!(data_mode, data_mode_temp)
        end
    end
    data_mode = data_mode[:,Not("rounded_pH")]

    NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", DataFrame)
    NIST_ESI = ()
    unique(Vector(NIST[:,:ION_MODE]))
    if ESI == -1
        ESI_name = "NEG"
        IE_raw = M2M4_neg
        NIST_ESI_ionmode = vcat(NIST[Vector(NIST[:,:ION_MODE]).=="N",:],NIST[Vector(NIST[:,:ION_MODE]).=="NEGATIVE",:])
        NIST_ESI = NIST_ESI_ionmode[((NIST_ESI_ionmode[:,:EXACT_MASS] - NIST_ESI_ionmode[:,:PRECURSOR_ION]) .< 1.01),:]
    elseif ESI == +1
        ESI_name = "POS"
        IE_raw = data_mode
        NIST_ESI_ionmode = vcat(NIST[Vector(NIST[:,:ION_MODE]).=="P",:],NIST[Vector(NIST[:,:ION_MODE]).=="POSITIVE",:])
        NIST_ESI = NIST_ESI_ionmode[((NIST_ESI_ionmode[:,:PRECURSOR_ION] - NIST_ESI_ionmode[:,:EXACT_MASS]) .< 1.01),:]
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    
    dataset = DataFrame()
    range_ = Int64.(round.(LinRange(1,size(NIST_ESI,1),(chunk_size+1))))
    for chunk = 1:chunk_size
        NIST_temp = NIST_df(NIST_ESI[range_[chunk]:range_[chunk+1],:])
        NIST_temp[!,:FORMULA] = convert.(String, NIST_temp[!,:FORMULA])
        select!(NIST_temp, Not([:RI_pred]))
        insertcols!(NIST_temp, 4, :pH_aq => -666.6)  # That's the pH column
        insertcols!(NIST_temp, 4, :logIE => -666.6)  # That's the IE column
        # For each Inchikey of the IE dataset, find the ones in all CNL datasets. The dataframe should contain all these entries (hcat)ed with the IE value
        for i = 1:size(IE_raw,1)
            key = IE_raw[i,"INCHIKEY"]           # That's the inchikey of compound i
            CNL_set_temp = NIST_temp[NIST_temp[:,"INCHIKEY"] .== key, :]# That's the CNLs of compound i
            CNL_set_temp[:,"pH_aq"] .= IE_raw[i,"pH.aq."]        # That's the pH of compound i
            CNL_set_temp[:,"logIE"] .= IE_raw[i,"unified_IEs"]        # That's the IE of compound i
            append!(dataset, CNL_set_temp)
            if (i % 500)==0
                println("NIST part $chunk/$(chunk_size): Compound $i/$(size(IE_raw,1)) - Mode $mode")
            end
        end
    end
    return dataset
end

CNL_IE_unified_zero_min = create_CNLIE_dataset_NIST_Unified_mode("min", HiValSetToZero=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero_min.csv", CNL_IE_unified_zero_min)
CNL_IE_unified_zero_max = create_CNLIE_dataset_NIST_Unified_mode("max", HiValSetToZero=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero_max.csv", CNL_IE_unified_zero_max)
CNL_IE_unified_zero_mean = create_CNLIE_dataset_NIST_Unified_mode("mean", HiValSetToZero=true)
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero_mean.csv", CNL_IE_unified_zero_mean)


