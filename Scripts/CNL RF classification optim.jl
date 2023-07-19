## import packages ##
using ScikitLearn
using BSON
using StatsPlots
using Statistics
using DataFrames
using CSV
using ScikitLearn.CrossValidation: cross_val_score
using ScikitLearn.CrossValidation: train_test_split
using PyCall
using Conda
using Random
using LinearAlgebra
using StatsBase
using JLD
@sk_import ensemble: RandomForestClassifier


#function Stratified_CNL_model_wFiltering_wConsensus_TestOnlyFiltered(ESI; consensus_threshold, allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, removeminusones::Bool=false, consensus_algorithm="replacement")
    function leverage_dist(unique_comps_fps, Norman)
        z = pinv(transpose(unique_comps_fps) * unique_comps_fps)
        lev = zeros(size(Norman,1))
        for j = 1:size(Norman,1)
            x = Norman[j,:]
            lev[j] = transpose(x) * z * x
            println("Leverage: $j/$(size(Norman,1))")
        end
        return lev
    end
    function split_classes(ESI, classes; random_state::Int=random_seed, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
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
                #println(j)
            end
            return lev
        end
            
        AD = leverage_dist(unique_comps_fps,unique_comps_fps)  
        inchi_train = []   
        inchi_test = []   
        try
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state,stratify = round.(AD,digits = 1))  
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)  
        end  
    
        return inchi_train, inchi_test
    end

    reg = []
    if ESI == -1
        ESI_name = "neg"
        ESI_symbol = "-"
        reg = cat.CatBoostRegressor(n_estimators=1600, learning_rate=0.15, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=10, verbose=false)
    elseif ESI == 1
        ESI_name = "pos"
        ESI_symbol = "+"
        reg = cat.CatBoostRegressor(n_estimators=1400, learning_rate=0.01, random_seed=3, grow_policy=:Lossguide, min_data_in_leaf=4, verbose=false)
    else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
    end

    # Load data files
    amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
    data_whole_raw = vcat(vcat(vcat(norman, amide), MB), NIST)

    # Scaling
    data_whole_raw[:,:MONOISOMASS] .= (data_whole_raw[:,:MONOISOMASS]) ./ 1000
    data_whole_raw[:,:pH_aq] .= (data_whole_raw[:,:pH_aq]) ./ 14

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
    
    # Split
    variables_df = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
    variables = Matrix(variables_df)

    # Stratified splitting (Test set)
    classes = unique(data_whole_filtered[:,:INCHIKEY])
    train_set_inchikeys, test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

    train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
    test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])

    train = data_whole_filtered[train_set_indices,:]
    X_test = variables[test_set_indices,:]
    y_test = data_whole_filtered[test_set_indices,:logIE]

    # Consensus spectra (replacing all spectra with one consensus spectrum)
    if (consensus_threshold > 0) && consensus_algorithm == "replacement"
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
            if (i % 250) == 0
            println("$i/$(size(data_info_unique,1))")
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
            if (i % 250) == 0
            println("$i/$(size(data_info_unique,1))")
            end
        end
    else
        data_whole = train
    end

    # Filtering again based on minimum number of CNLs per spectrum
    no_CNLs_per_spectrum_afterConsensusFiltering = (count(==(1), Matrix(data_whole[:,9:end]) ; dims=2))[:]
    data_train = data_whole[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum_afterConsensusFiltering),:]

    # Splitting (Train set)
    X_train = Matrix(hcat(data_train[:,:pH_aq],data_train[:,8:end]))
    y_train =  data_train[:,:logIE]

    ## Regression ##
    ScikitLearn.fit!(reg, X_train, y_train)
    importance = sort(reg.feature_importances_, rev=true)
    importance_index = sortperm(reg.feature_importances_, rev=true)
    significant_columns = importance_index[importance .>=1]

    z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
    z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
    z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
    z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
    z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
    z6 = z4 - y_train      # Train set residual
    z7 = z5 - y_test        # Test set residual


    # Plots
    if allowplots == true
        p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
        scatter!(y_test,z5,label="Test set", color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
        annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
        annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
        p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)
        p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange)
        plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)

        p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_Consensus_$ESI_name.png")
        end

        p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, label="Test set",color=:orange)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
        p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma
        p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma

        p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_Consensus_$ESI_name.png")
        end
        display(p123)
        display(p456)
        if showph == true           
            plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
            scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_Consensus_$ESI_name.png")
            end

            plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
    
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_Consensus_$ESI_name.png")
            end
            display(plot_pH)
            display(plot_pH_res)
        end
    end
    return z1,z2,z3,z4,z5,z6,z7
#end

# 15-Jun-2023

function Stratified_CNL_Classification_wFiltering_wConsensus_TestOnlyFiltered(ESI; consensus_threshold, trees::Int, leaf::Int,allowplots::Bool=false, allowsave::Bool=false, showph::Bool=false, split_size::Float64=0.2, min_CNLs::Int, random_seed::Int, removeminusones::Bool=false, consensus_algorithm="replacement")
    function split_classes(ESI, classes; random_state::Int=random_seed, split_size::Float64=0.2)
        if ESI == -1
            ESI_name = "neg"
        elseif ESI == 1
            ESI_name = "pos"
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end
        FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Fingerprints\\padel_M2M4_$(ESI_name)_12_w_inchikey.csv", DataFrame)
        #classes = unique(FP[:,:INCHIKEY])
    
        # group the grades by student and calculate the mean grade for each student
        average_logIE_all = combine(groupby(FP[:,[:INCHIKEY,:logIE]], :INCHIKEY), :logIE => mean => :logIE_mean)
        average_logIE_filtered = filter(row -> row.INCHIKEY in classes, average_logIE_all)
        AD = round.((round.(average_logIE_filtered[:,:logIE_mean],digits = 0) .+0.01), digits =0)
        map = (sort(countmap(AD)))
        for small_group in sort(unique(AD))[values(map) .< 3]
            if small_group < 0
                AD[AD .== small_group] .= small_group+1
            elseif small_group >= 0
                AD[AD .== small_group] .= small_group-1
            end
        end

        inchi_train = []   
        inchi_test = []
        #split_size=0.2
        # random_state=1312
        try
            inchi_train, inchi_test = train_test_split(average_logIE_filtered[:,:INCHIKEY], test_size=split_size, random_state=random_state,stratify = AD)  
        catch
            inchi_train, inchi_test = train_test_split(classes, test_size=split_size, random_state=random_state)  
        end  
    
        return inchi_train, inchi_test
    end
        reg = []
        #ESI=+1
        #trees=tree
        if ESI == -1
            ESI_name = "neg"
            ESI_symbol = "-"
            threshold_up = 1.5
            threshold_down = 0    
            reg = RandomForestClassifier(n_estimators= trees, min_samples_leaf = leaf, oob_score=true, n_jobs=-1, random_state=random_seed)
        elseif ESI == 1
            ESI_name = "pos"
            ESI_symbol = "+"
            threshold_up = 3
            threshold_down = 1   
            reg = RandomForestClassifier(n_estimators= trees, min_samples_leaf = leaf, oob_score=true, n_jobs=-1, random_state=random_seed)
        else error("Set ESI to -1 or +1 for ESI- and ESI+ accordingly")
        end

        # Load data files
        amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
        norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
        MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
        NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
        data_whole_raw = vcat(vcat(vcat(norman, amide), MB), NIST)

        # Remove any NaN values
        deleteat!(data_whole_raw, findall(x->x.==1,isnan.(data_whole_raw[:,:MONOISOMASS])))

        # Scaling
        data_whole_raw[:,:MONOISOMASS] .= (data_whole_raw[:,:MONOISOMASS]) ./ 1000
        data_whole_raw[:,:pH_aq] .= (data_whole_raw[:,:pH_aq]) ./ 14

        # Removing the -1s
        if removeminusones == true
            for j = 1:size(data_whole_raw,1)
                data_whole_raw[j,[Vector(data_whole_raw[j,:]).==-1][1]] .= 0
                if (j % 50000) == 0
                    println("Removing -1: $j/ $(size(data_whole_raw,1))")
                end
            end
        end
        # min_CNLs = 1
        # Filtering based on minimum number of CNLs per spectrum
        no_CNLs_per_spectrum = (count(==(1), Matrix(data_whole_raw[:,9:end]) ; dims=2))[:]
        data_whole_filtered = data_whole_raw[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum),:]
        
        # Creating the groups
        boxplot(data_whole_filtered[:,:logIE])#,bins=200)
        #high_group
        data_whole_filtered[data_whole_filtered[:,:logIE] .>= threshold_up,:Column1] .= "High"
        #med_group
        data_whole_filtered[(data_whole_filtered[:,:logIE] .< threshold_up) .&& (data_whole_filtered[:,:logIE] .> threshold_down),:Column1] .= "Medium"
        #low_group
        data_whole_filtered[data_whole_filtered[:,:logIE] .<= threshold_down,:Column1] .= "Low"
        countmap(data_whole_filtered[:,:Column1])


        scatter(data_whole_filtered[:,:logIE], group = data_whole_filtered[:,:Column1])
        
        # Split
        variables_df = hcat(data_whole_filtered[:,:pH_aq],data_whole_filtered[:,8:end])
        variables = Matrix(variables_df)

        # Stratified splitting (Test set)
        #random_seed = 1312
        #consensus_threshold = 0.3
        classes = unique(data_whole_filtered[:,:INCHIKEY])
        train_set_inchikeys, test_set_inchikeys = split_classes(ESI, classes; random_state=random_seed)

        train_set_indices = findall(x -> x in train_set_inchikeys, data_whole_filtered[:,:INCHIKEY])
        test_set_indices = findall(x -> x in test_set_inchikeys, data_whole_filtered[:,:INCHIKEY])

        train = data_whole_filtered[train_set_indices,:]
        X_test = variables[test_set_indices,:]
        y_test = data_whole_filtered[test_set_indices,:Column1]

        # Consensus spectra (replacing all spectra with one consensus spectrum)
        if (consensus_threshold > 0) && consensus_algorithm == "replacement"
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
                if (i % 250) == 0
                println("$i/$(size(data_info_unique,1))")
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
                if (i % 250) == 0
                println("$i/$(size(data_info_unique,1))")
                end
            end
        else
            data_whole = train
        end

        # Filtering again based on minimum number of CNLs per spectrum
        no_CNLs_per_spectrum_afterConsensusFiltering = (count(==(1), Matrix(data_whole[:,9:end]) ; dims=2))[:]
        data_train = data_whole[findall(x -> x .>= min_CNLs, no_CNLs_per_spectrum_afterConsensusFiltering),:]

        # Splitting (Train set)
        X_train = Matrix(hcat(data_train[:,:pH_aq],data_train[:,8:end]))
        y_train =  data_train[:,:Column1]

        ## Regression ##
        # reg= RandomForestClassifier()
        ScikitLearn.fit!(reg, X_train, y_train)
        importance = 100 .* sort(reg.feature_importances_, rev=true)
        importance_index = sortperm(reg.feature_importances_, rev=true)
        significant_columns = importance_index[importance .>=1]

        z1 = names(variables_df[:,:])[significant_columns]   # Most important descriptors
        z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
        z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
        z4 = ScikitLearn.predict(reg, X_train)     # y_hat_train
        z5 = ScikitLearn.predict(reg, X_test)   # y_hat_test
        #z6 = z4 - y_train      # Train set residual
        #z7 = z5 - y_test        # Test set residual


        # Plots
        if allowplots == true
            p1 = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)")
            scatter!(y_test,z5,label="Test set", color=:orange)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],label="1:1 line",width=2)
            annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
            annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)
            p2 = scatter(y_train,z4,legend=false,ticks=false,color = :magenta)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)
            p3 = scatter(y_test,z5,legend=false,ticks=false, color=:orange)
            plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],width=2)

            p123 = plot(p1,p2,p3,layout= @layout [a{0.7w} [b; c]])
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_Consensus_$ESI_name.png")
            end

            p4 = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals", color = :magenta, xlabel = "Experimental log(IE)", ylabel = "Residual")
            scatter!(y_test,z7, label="Test set",color=:orange)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="pred = exp",width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
            p5 = scatter(y_train,z6, legend=false, ticks=false, color = :magenta)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma
            p6 = scatter(y_test,z7, label="Test set",color=:orange,legend=false, ticks=false)
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],width=2) # 1:1 line
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],linecolor ="grey",width=2) # +3 sigma
            plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],linecolor ="grey",width=2) #-3 sigma

            p456 = plot(p4,p5,p6,layout= @layout [a{0.7w} [b; c]])
            if allowsave == true
                savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_Consensus_$ESI_name.png")
            end
            display(p123)
            display(p456)
            if showph == true           
                plot_pH = scatter(y_train,z4,label="Training set", legend=:best, title = "ESI$(ESI_symbol) IEs from CNL", markershape = :circle, marker_z = X_train[:,1] , xlabel = "Experimental log(IE)", ylabel = "Predicted log(IE)",color=:jet)
                scatter!(y_test,z5,label="Test set", marker_z = X_test[:,1] , markershape = :rect,color=:jet)
                plot!([minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))],[minimum(vcat(y_train,y_test)),maximum(vcat(y_train,y_test))], label="1:1 line",width=2)
                annotate!(maximum(vcat(y_train,y_test)),0.8+minimum(vcat(y_train,y_test)),latexstring("Training: R^2=$(round(z2, digits=3))"),:right)
                annotate!(maximum(vcat(y_train,y_test)),0.3+minimum(vcat(y_train,y_test)),latexstring("Test: R^2=$(round(z3, digits=3))"),:right)  
                    if allowsave == true
                    savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_Consensus_$ESI_name.png")
                end

                plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
                scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
                plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
                plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
                plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma
        
                if allowsave == true
                    savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_Consensus_$ESI_name.png")
                end
                display(plot_pH)
                display(plot_pH_res)
            end
        end
        return z1,z2,z3,z4,z5
end


iterations = 10
leaf_r = collect(4:2:10)
tree_r = vcat([50,100], vcat(collect(200:200:800),(collect(1000:200:1600))))
random_seed_r = collect(1:1:3)
min_CNLs_R = collect(1:10)
removeminusones_r = [true,false]
consensus_threshold_r = [0.1,0.2,0.25]
consensus_algorithm_r = ["replacement", "removal"]

results_for_filtered_data = DataFrame(min_CNLs=Int.(zeros(iterations*length(min_CNLs_R))),accuracy_train=zeros(iterations*length(min_CNLs_R)),accuracy_test=zeros(iterations*length(min_CNLs_R)),leaf=Int.(zeros(iterations*length(min_CNLs_R))), tree=Int.(zeros(iterations*length(min_CNLs_R))), random_seed=Int.(zeros(iterations*length(min_CNLs_R))), removeminusones=Int.(zeros(iterations*length(min_CNLs_R))), consensus_threshold=zeros(iterations*length(min_CNLs_R)), consensus_algorithm= string.(zeros(iterations*length(min_CNLs_R))))

for i = 1:length(min_CNLs_R)
    for itr = 1:iterations
        leaf = rand(leaf_r)
        tree = rand(tree_r)
        random_seed = rand(random_seed_r)
        removeminusones = rand(removeminusones_r)
        consensus_threshold = rand(consensus_threshold_r)
        consensus_algorithm = rand(consensus_algorithm_r)

        z1, accuracy_tr ,accuracy_te ,z4,z5 = Stratified_CNL_Classification_wFiltering_wConsensus_TestOnlyFiltered(+1, consensus_threshold=consensus_threshold, removeminusones=removeminusones, consensus_algorithm=consensus_algorithm, min_CNLs=min_CNLs_R[i], random_seed=random_seed, trees=tree, leaf=leaf)

        index_df = ((i-1)*iterations) + itr
        results_for_filtered_data[index_df,:min_CNLs] = min_CNLs_R[i]
        results_for_filtered_data[index_df,:accuracy_train] = accuracy_tr
        results_for_filtered_data[index_df,:accuracy_test] = accuracy_te
        results_for_filtered_data[index_df,:leaf] = leaf
        results_for_filtered_data[index_df,:tree] = tree
        results_for_filtered_data[index_df,:random_seed] = random_seed
        results_for_filtered_data[index_df,:removeminusones] = removeminusones
        results_for_filtered_data[index_df,:consensus_threshold] = consensus_threshold
        results_for_filtered_data[index_df,:consensus_algorithm] = consensus_algorithm
        println("$index_df/$(iterations*(length(min_CNLs_R))) Done!")
    end
end

CSV.write("/media/emcms/Elements SE/Data Alex/IE_prediction/Figures/")
#=
X_train = rand(1:10:100,5,2)
y_train = rand(["Yes","No"],5)
X_test = rand(1:20,5,2)
y_test = rand(["Yes","No"],5)
reg = RandomForestClassifier(oob_score=true, n_jobs=-1, random_state=1312)
ScikitLearn.fit!(reg, X_train, y_train)

z4 = [ScikitLearn.predict(reg, X_train) y_train]     # y_hat_train
z5 = [ScikitLearn.predict(reg, X_test) y_test]   # y_hat_test

z2 = ScikitLearn.score(reg, X_train, y_train)   # Train set accuracy
z3 = ScikitLearn.score(reg, X_test, y_test)      # Test set accuracy
=#