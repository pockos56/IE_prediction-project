## import packages ##
using ScikitLearn, ProgressBars, Statistics, DataFrames, CSV, PyCall, Conda, LaTeXStrings, LinearAlgebra, Random, StatsBase
using ScikitLearn.CrossValidation: train_test_split
cat = pyimport("catboost")
jblb = pyimport("joblib")



function Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(; consensus_threshold, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, tree::Int, learn_rate, leaf::Int, removeminusones::Bool=false, consensus_algorithm="replacement")
    function split_classes(classes; random_state::Int=random_seed)
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Internal_pubchem_FPs_dict.csv", DataFrame)
        indices = [Int(findfirst(x->x .== classes[i], FP[:,:INCHIKEY])) for i in 1:length(classes)]
        unique_comps_fps = Matrix(FP[indices,2:end])
    
        function leverage_dist(unique_comps_fps, Norman)
            z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
            lev = zeros(size(Norman,1))
            for j = 1:size(Norman,1)
                lev[j] = transpose(Norman[j,:]) * z * Norman[j,:]
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)  
        inchi_train = []   
        inchi_test = []   #random_state=1
        try
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))  
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)  
        end  
    
        return inchi_train, inchi_test
    end

    # Load data files
    data_whole_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\CNL-IE\\CNL_IE_unified_zero.csv", DataFrame)
    # Scaling
    data_whole_raw[:,"MONOISOMASS"] .= (data_whole_raw[:,"MONOISOMASS"]) ./ 1000
    data_whole_raw[:,"pH_aq"] .= (data_whole_raw[:,"pH_aq"]) ./ 14
    # Removing the -1s
    if removeminusones == true
        for j = 1:size(data_whole_raw,1)
            data_whole_raw[j,[Vector(data_whole_raw[j,:]).==-1][1]] .= 0
            if (j % 50000) == 0
                println("Removing -1: $j/ $(size(data_whole_raw,1))")
            end
        end
    end

    # Filtering based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum = (count(==(1), Matrix(data_whole_raw[:,9:end]) ; dims=2))[:]
    data_whole_filtered = data_whole_raw[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum),:]
    #min_CNLs=3
    # Split
    variables_df = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df)

    # Stratified splitting (Test set)
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(classes; random_state=random_seed)
    #random_seed=3

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    train = data_whole_filtered[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_test = data_whole_filtered[test_set_indices,:logIE]

    # Consensus spectra (replacing all spectra with one consensus spectrum)
    if (consensus_threshold > 0) && consensus_algorithm == "replacement"
        data_info_unique = unique(train[:,2:8])
        data_whole = DataFrame()
        for i = 1:size(data_info_unique,1)
            group_members = train[all(Matrix(train[:,2:8] .== DataFrame(data_info_unique[i,:])),dims=2)[:],:]
            min_freq = Int(floor(size(group_members,1)*consensus_threshold))
            if size(group_members,1) >= (3+min_freq)
                CNL_possible_indices = []
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    append!(CNL_possible_indices,ones_indices)                    
                end
                freq_dict = sort(countmap(CNL_possible_indices), rev=true)
                most_freq_CNLs = collect(keys(freq_dict))[collect(values(freq_dict) .>= min_freq)]

                consensus_spectrum_info = DataFrame(group_members[1,1:8])
                consensus_spectrum_cnl = DataFrame(group_members[1,9:end])
                consensus_spectrum_cnl[:,(Matrix(consensus_spectrum_cnl).==1)[:]] .= 0
                consensus_spectrum_i = hcat(consensus_spectrum_info, consensus_spectrum_cnl)
                consensus_spectrum_i[:,most_freq_CNLs] .= 1
                append!(data_whole, consensus_spectrum_i)
            else
                append!(data_whole, group_members)
            end
            if (i % 5000) == 0
            println("Replacing with consensus spectrum: $i/$(size(data_info_unique,1))")
            end
        end
    # Consensus spectra (removing fragments that are not present in the consensus spectrum)
    elseif (consensus_threshold > 0) && consensus_algorithm == "removal"
        data_info = train[:,2:8]
        data_info_unique = unique(data_info)
        data_whole = DataFrame()
        for i = 1:size(data_info_unique,1)
            group_members = train[all(Matrix(train[:,2:8] .== DataFrame(data_info_unique[i,:])),dims=2)[:],:]
            min_freq = Int(floor(size(group_members,1)*consensus_threshold))

            if size(group_members,1) >= (3+min_freq)
                CNL_possible_indices = []
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    append!(CNL_possible_indices,ones_indices)                    
                end
                freq_dict = sort(countmap(CNL_possible_indices), rev=true)
                most_freq_CNLs = Int.(collect(keys(freq_dict))[collect(values(freq_dict) .>= min_freq)] ) #These are the acceptable CNLs, everything else should be removed
                
                for j = 1:size(group_members,1)
                    ones_indices = findall(x -> x .== 1, Matrix(group_members)[j,:])
                    ones_to_be_removed = setdiff(ones_indices, most_freq_CNLs)
                    group_members[j,ones_to_be_removed] .= 0
                end
                #consensus_spectrum_info = DataFrame(group_members[1,1:8])
                #consensus_spectrum_cnl = DataFrame(group_members[1,9:end])
                #consensus_spectrum_cnl[:,(Matrix(consensus_spectrum_cnl).==1)[:]] .= 0
                #consensus_spectrum_i = hcat(consensus_spectrum_info, consensus_spectrum_cnl)
                #consensus_spectrum_i[:,most_freq_CNLs] .= 1
                append!(data_whole, group_members)
            else
                append!(data_whole, group_members)
            end
            if (i % 5000) == 0
            println("Removing uncommon fragments: $i/$(size(data_info_unique,1))")
            end
        end
    else
        data_whole = train
    end

    # Filtering again based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum_afterConsensusFiltering = (count(==(1), Matrix(data_whole[:,9:end]) ; dims=2))[:]
    data_train = data_whole[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum_afterConsensusFiltering),:]

    # Splitting (Train set)
    X_train = Matrix(hcat(data_train[:,"pH_aq"],data_train[:,8:end]))
    y_train =  data_train[:,:logIE]

    ## Regression ##
    reg = cat.CatBoostRegressor(n_estimators=tree, learning_rate=learn_rate, random_seed=random_seed, grow_policy=:Lossguide, min_data_in_leaf=leaf, verbose=false)
    ScikitLearn.fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    significant_columns = sortperm(reg.feature_importances_, rev=true)[importance .>=1]

    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual

    return z1,z2,z3,z4,z5,z6,z7
end

iterations = 2
leaf_r = collect(4:2:10)
tree_r = vcat(collect(200:200:800),(collect(1000:200:1600)))
random_seed_r = collect(1:3)
learn_r = [0.01,0.05,0.1,0.15,0.2]
consensus_threshold_r = [0, 0.1, 0.2, 0.25]
min_CNLs_r = collect(0:1:3)
consensus_algorithm_r = ["removal","replacement"]
removeminusones_r = [true,false]
results = DataFrame(min_CNLs=Int.(zeros(iterations)),consensus_thres=zeros(iterations),consensus_alg = string.(missings(iterations)),removeminusones_df =Int.(zeros(iterations)),random_seed=Int.(zeros(iterations)),tree=Int.(zeros(iterations)),leaf=Int.(zeros(iterations)),learn_rate=zeros(iterations), accuracy_train=zeros(iterations),accuracy_test=zeros(iterations))

for itr in ProgressBar(1:iterations)
    min_CNLs_itr = rand(min_CNLs_r)
    leaf_itr = rand(leaf_r)
    tree_itr = rand(tree_r)
    random_seed_itr = rand(random_seed_r)
    learn_rate_itr = rand(learn_r)
    consensus_threshold_itr = rand(consensus_threshold_r)
    consensus_algorithm_itr = rand(consensus_algorithm_r)
    removeminusones_itr = rand(removeminusones_r)

    z1, accuracy_tr ,accuracy_te ,z4,z5,z6,z7 = Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(consensus_threshold=consensus_threshold_itr, consensus_algorithm = consensus_algorithm_itr,removeminusones = removeminusones_itr, min_CNLs=min_CNLs_itr, random_seed=random_seed_itr, tree=tree_itr, learn_rate=learn_rate_itr, leaf=leaf_itr)

    results[itr,"min_CNLs"] = min_CNLs_itr
    results[itr,"consensus_thres"] = consensus_threshold_itr
    results[itr,"consensus_alg"] = consensus_algorithm_itr
    results[itr,"removeminusones_df"] = removeminusones_itr        
    results[itr,"random_seed"] = random_seed_itr
    results[itr,"tree"] = tree_itr
    results[itr,"leaf"] = leaf_itr
    results[itr,"learn_rate"] = learn_rate_itr
    results[itr,"accuracy_train"] = accuracy_tr
    results[itr,"accuracy_test"] = accuracy_te

    CSV.write("", results)
end
CSV.write("", results)
