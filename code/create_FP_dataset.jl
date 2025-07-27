########################################################################
## Goal: Create the FP dataset

using CSV, DataFrames, Statistics, ProgressBars

# Compute fingerprints
path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Fingerprints\\"
for i in ProgressBar(1:16)
    FP_mean = DataFrame()
    FP = CSV.read(path*"FP$i.csv", DataFrame)
    insertcols!(FP, 5, "rounded_pH" => round.(FP[:,"pH.aq."]))
    for unique_inchikey in ProgressBar(unique(FP[:,"INCHIKEY"]))
        FP_comp = FP[findall(x->x .== unique_inchikey, FP[:,"INCHIKEY"]),:]
        for unique_pH in unique(FP_comp[:,"rounded_pH"])
            FP_comp_pH = FP_comp[findall(x->x .== unique_pH, FP_comp[:,"rounded_pH"]),:]
            # FP Mean
            FP_mean_temp = deepcopy(DataFrame(FP_comp_pH[1,:]))
            FP_mean_temp[1,"unified_IEs"] = mean(FP_comp_pH[:,"unified_IEs"])
            FP_mean_temp[1,"pH.aq."] = mean(FP_comp_pH[:,"pH.aq."])
            FP_mean = append!(FP_mean, FP_mean_temp)
        end
    end
    CSV.write(path*"FP$(i)_mean.csv",FP_mean[:,Not("rounded_pH")])
end
