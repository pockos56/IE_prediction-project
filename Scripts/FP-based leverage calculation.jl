## import packages ##
using ScikitLearn
using BSON
using Plots
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
using Random
using LinearAlgebra
cat = pyimport("catboost")
using Distances


function leverage_calc(ESI;random_state::Int=1312)
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)

    #New splitting
    # Make reference mean
    classes = unique(FP[:,:INCHIKEY])
    indices = Int.(zeros(length(classes)))
    for i = 1:length(classes)
        inchi_temp = classes[i]
        indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
    end
    unique_comps_fps = Matrix(FP[indices,9:end])

    # Calculate projection-based leverage
    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println(j)
        end
        return lev
    end
    
    function cityblock_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
            println(j)
        end
        return lev
    end

    leverages_projection = leverage_dist(unique_comps_fps, Matrix(FP[:,9:end]))
    leverages_manhattan = cityblock_dist(unique_comps_fps, Matrix(FP[:,9:end]))
    return leverages_projection, leverages_manhattan
end

levs_proj_neg, levs_manh_neg = leverage_calc(-1)
levs_proj_pos, levs_manh_pos = leverage_calc(+1)




histogram(levs_proj_neg,bins=50)
histogram(levs_manh_neg,bins=50)
histogram(levs_proj_pos,bins=50)
histogram(levs_manh_pos,bins=50)

function split_classes(ESI; random_state::Int=1312)
    if ESI == -1
        ESI_name = "neg"
    elseif ESI == 1
        ESI_name = "pos"
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end
    FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
    classes = unique(FP[:,:INCHIKEY])
    indices = Int.(zeros(length(classes)))
    for i = 1:length(classes)
        inchi_temp = classes[i]
        indices[i] = Int(findfirst(x->x .== inchi_temp, FP[:,:INCHIKEY]))
    end
    unique_comps_fps = Matrix(FP[indices,9:end])

    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println(j)
        end
        return lev
    end
    
    function cityblock_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            lev[j] = sqrt(sum(sqrt.(colwise(cityblock,Norman[j,:],z))))
            println(j)
        end
        return lev
    end

    AD = leverage_dist(unique_comps_fps,unique_comps_fps)
    #AD_cityblock = cityblock_dist(unique_comps_fps,unique_comps_fps)       # Implement in the future
    
    inchi_train, inchi_test = train_test_split(classes, test_size=0.20, random_state=random_state,stratify = round.(AD,digits = 1))

    return inchi_train, inchi_test
end

X_train_neg, X_test_neg = split_classes(-1)
X_train_pos, X_test_pos = split_classes(+1)
