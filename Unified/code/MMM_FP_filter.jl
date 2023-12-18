using CSV, DataFrames, Statistics, ProgressBars

path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\"
for i in ProgressBar(1:16)
    FP_min = DataFrame()
    FP_max = DataFrame()
    FP_mean = DataFrame()
    FP = CSV.read(path*"FP$i.csv", DataFrame)
    insertcols!(FP, 5, "rounded_pH" => round.(FP[:,"pH.aq."]))
    for unique_inchikey in ProgressBar(unique(FP[:,"INCHIKEY"]))
        FP_comp = FP[findall(x->x .== unique_inchikey, FP[:,"INCHIKEY"]),:]
        for unique_pH in unique(FP_comp[:,"rounded_pH"])
            FP_comp_pH = FP_comp[findall(x->x .== unique_pH, FP_comp[:,"rounded_pH"]),:]
            # FP Minimum
            FP_min_temp = deepcopy(DataFrame(FP_comp_pH[1,:]))
            FP_min_temp[1,"unified_IEs"] = minimum(FP_comp_pH[:,"unified_IEs"])
            FP_min_temp[1,"pH.aq."] = mean(FP_comp_pH[:,"pH.aq."])
            FP_min = append!(FP_min, FP_min_temp)
            # FP Maximum
            FP_max_temp = deepcopy(DataFrame(FP_comp_pH[1,:]))
            FP_max_temp[1,"unified_IEs"] = maximum(FP_comp_pH[:,"unified_IEs"])
            FP_max_temp[1,"pH.aq."] = mean(FP_comp_pH[:,"pH.aq."])
            FP_max = append!(FP_max, FP_max_temp)
            # FP Mean
            FP_mean_temp = deepcopy(DataFrame(FP_comp_pH[1,:]))
            FP_mean_temp[1,"unified_IEs"] = mean(FP_comp_pH[:,"unified_IEs"])
            FP_mean_temp[1,"pH.aq."] = mean(FP_comp_pH[:,"pH.aq."])
            FP_mean = append!(FP_mean, FP_mean_temp)
        end
    end
    CSV.write(path*"FP$(i)_min.csv",FP_min[:,Not("rounded_pH")])
    CSV.write(path*"FP$(i)_max.csv",FP_max[:,Not("rounded_pH")])
    CSV.write(path*"FP$(i)_mean.csv",FP_mean[:,Not("rounded_pH")])
end
