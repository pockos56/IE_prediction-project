## import packages ##
using ScikitLearn
using CSV
using Statistics
using DataFrames
using PyCall
using Conda
using DataStructures
using PkgTemplates
jblb = pyimport("joblib")
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")
#using BSON
#using CSV
#using Plots

# Loading models
CNL_reg_neg = jblb.load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_reg_neg.joblib")
CNL_reg_pos = jblb.load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\CNL_reg_pos.joblib")
FP_reg_neg = jblb.load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_reg_neg.joblib")
FP_reg_pos = jblb.load("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Models\\FP_reg_pos.joblib")
best_CNLs_neg = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLmax_Hadducts_neg.CSV", DataFrame)[1:500,1]
best_CNLs_pos = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNLmax_Hadducts_pos.CSV", DataFrame)[1:500,1]

# Creating the two functions
function IE_from_InChIKey(INCHIKEY::String, ESI_mode::String, pH)
    if pH > 14 || pH < 0 
        error("Set pH to a valid value between 0 and 14")
    end

    if ESI_mode == "negative" || ESI_mode == "neg"
        reg = FP_reg_neg
    elseif ESI_mode == "positive" || ESI_mode == "pos"
        reg = FP_reg_pos
    else error("ESI_mode should be set to positive or negative")
    end

    fingerprint = DataFrame()
    find_comp = pcp.get_compounds(INCHIKEY, "inchikey")[1]
    fingerprint = Int.(parse.(Float64,Matrix(DataFrame(pd.from_smiles(find_comp.isomeric_smiles,fingerprints=true, descriptors=false)))))
    if size(fingerprint,2) != 780
        error("Wrong type of fingerprints was calculated. Check if the descriptors.xml file is present and set to 2DAPC")
    end

    input = Matrix(hcat(pH, fingerprint))[1,:]
    IE_pred = predict(reg, input)
    return IE_pred        
end

function IE_from_SMILES(SMILES::String, ESI_mode::String, pH)
    if pH > 14 || pH < 0 
        error("Set pH to a valid value between 0 and 14")
    end

    if ESI_mode == "negative" || ESI_mode == "neg"
        reg = FP_reg_neg
    elseif ESI_mode == "positive" || ESI_mode == "pos"
        reg = FP_reg_pos
    else error("ESI_mode should be set to positive or negative")
    end

    fingerprint = DataFrame()
    fingerprint = Int.(parse.(Float64,Matrix(DataFrame(pd.from_smiles(SMILES,fingerprints=true, descriptors=false)))))
    if size(fingerprint,2) != 780
        error("Wrong type of fingerprints was calculated. Check if the descriptors.xml file is present and set to 2DAPC")
    end

    input = Matrix(hcat(pH, fingerprint))[1,:]
    IE_pred = predict(reg, input)
    return IE_pred        
end

function IE_from_MS(fragments_list::Vector, precursor_ion_mz::Float64, ESI_mode::String, pH)
    function mz_to_fingerprint(CNL_vec, precursor; threshold=0.02)
        # Convert string to vector{float64}
        #df_input = DataFrame(df_row)
        #vector_input = Matrix(df_input)[:]
        #CNL_vec = Float64.(eval(Meta.parse.(names(df_input[:,findall(x -> x .== 1, vector_input)]))))
        #precursor = Float64.(eval(Meta.parse.(names(df_input[:,findfirst(x -> x .== -1, vector_input):1+findfirst(x -> x .== -1, vector_input)]))))[1]
        # Define the range and step size for the fingerprint
        fingerprint_range_raw = []
        if ESI_mode == "negative" || ESI_mode == "neg"
            fingerprint_range_raw = round.(best_CNLs_neg, digits=2)
        elseif ESI_mode == "positive" || ESI_mode == "pos"
            fingerprint_range_raw = round.(best_CNLs_pos, digits=2)
        else error("ESI_mode should be set to positive or negative")
        end
        
        distance_threshold = threshold - 0.01
        fingerprint_range = fingerprint_range_raw[1:2]
        for i = 3:length(fingerprint_range_raw)
            distance = round.(abs.(fingerprint_range_raw[i] .- fingerprint_range),digits=3)
            if all(distance .> distance_threshold)
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
        fingerprint_df[:,idx_precursor+1:end] .= -1
        return fingerprint_df
    end
    if pH > 14 || pH < 0 
        error("Set pH to a valid value between 0 and 14")
    end

    if ESI_mode == "negative" || ESI_mode == "neg"
        reg = CNL_reg_neg
    elseif ESI_mode == "positive" || ESI_mode == "pos"
        reg = CNL_reg_pos
    else error("ESI_mode should be set to positive or negative")
    end

    CNLs = round.(precursor_ion_mz .- fragments_list, digits=2)     # CNL calculation based on precursor ion mass and fragments
    df = Matrix(mz_to_fingerprint(CNLs, precursor_ion_mz))          # Transforming the CNLs to a fingerprint-like format with a binary vector of the most probable CNLs
    df_all = hcat(hcat(pH, precursor_ion_mz), df)                   # Add the required 'pH' and 'precursor ion mass' variables

    # Scaling
    df_all[:,1] = (df_all[:,1]) ./ 14          # pH Scaling
    df_all[:,2] = (df_all[:,2]) ./ 1000        # MONOISOMASS Scaling

    # Removing minus ones for ESI- model
    if ESI_mode == "negative" || ESI_mode == "neg"      
        df_all[1,[Vector(df_all[1,:]).==-1][1]] .= 0                # This condition should be true for the ESI- model
    end
    
    input = (df_all)[1,:]
    IE_pred = predict(reg, input)
    return IE_pred        
end

# Testing #TO DELETE#

    ESI_mode = "negative"
    pH = 6

    INCHIKEY = "AOJJSUZBOXZQNB-TZSSRYMLSA-N"
    IE_from_InChIKey(INCHIKEY, ESI_mode, 0)

    SMILES = "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O"
    IE_from_SMILES(SMILES, ESI_mode, pH)

    
    fragments_list = [33.11]

    precursor_ion_mz = 100.1
    IE_from_MS(fragments_list, precursor_ion_mz, "negative", 5)



#


# Creating the jl file
t= Template(;
    user="alex_",
    dir="C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Packages",
    authors=["Alexandros Nikolopoulos", "Env Modeling & Computational Mass Spec Lab (EMCMS)"],
    julia=v"1.8.5",
    plugins=[
        License(; name="MIT"),
        Git(; manifest=true, ssh=true),
        #TravisCI(),
        #Codecov(),
        AppVeyor(),
    ],
)
t("IE_prediction.jl")