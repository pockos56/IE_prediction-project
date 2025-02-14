using CSV, DataFrames, ProgressBars, Plots

function create_df_mode(data_mode::String; allowplots::Bool=false, allowsave::Bool=false)
    if data_mode == "min"
        df_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_min.csv", DataFrame)
        df_CNL = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_min.csv", DataFrame)
    elseif data_mode == "mean"
        df_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_mean.csv", DataFrame)
        df_CNL = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_mean.csv", DataFrame)
    elseif data_mode == "max"
        df_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_max.csv", DataFrame)
        df_CNL = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_max.csv", DataFrame)
    else error("Set data_mode to min, mean, or max")
    end

    df_mode = hcat(deepcopy(df_CNL), DataFrame("IE_hat_FP"=>-Inf*ones(size(df_CNL,1)), "class_FP"=>"TBD"))
    # Rounding the pH values
    df_mode[:,"pH_aq"] = round.(df_mode[:,"pH_aq"]; digits=3)
    df_FP[:,"pH_aq"] = round.(df_FP[:,"pH_aq"]; digits=3)
    failed_inchikeys = []
    for i in ProgressBar(1:size(df_mode,1)) 
        FP_same_inchikey = df_FP[findall(x->x .== df_mode[i,"INCHIKEY"], df_FP[:,"INCHIKEY"]),:]
        index_same_pH = findall(x->x .== df_mode[i,"pH_aq"], FP_same_inchikey[:,"pH_aq"])
        if length(index_same_pH) != 1
            failed_inchikeys = vcat(failed_inchikeys, df_mode[i,"INCHIKEY"])
        else
            df_mode[i,"IE_hat_FP"] = FP_same_inchikey[index_same_pH[1],"IE_hat_fp"]
            df_mode[i,"class_FP"] = FP_same_inchikey[index_same_pH[1],"class_fp"]
        end
    end

    # Filter non-matches
    deleteat!(df_mode, findall(x->x.==-Inf,df_mode[:,"IE_hat_FP"]))

    #Calculate residuals
    df_mode_residuals_FP = df_mode[:,"IE_hat_FP"] - df_mode[:,"IE"]
    df_mode_residuals_CNL = df_mode[:,"IE_hat_CNL"] - df_mode[:,"IE"]
    
    #Plots
    if allowplots
        p1 = scatter(df_mode[:,"IE_hat_FP"],df_mode[:,"IE_hat_CNL"],label=false, alpha=0.7, xlabel="Structure-based predicted logIE", ylabel="MS-based predicted logIE", legend=:best, c="magenta", xlims=(-0.7,5.7),ylims=(-0.3,5.3),dpi=300)
        plot!([minimum(df_mode[:,"IE_hat_FP"]),maximum(df_mode[:,"IE_hat_FP"])],[minimum(df_mode[:,"IE_hat_FP"]),maximum(df_mode[:,"IE_hat_FP"])],label="1:1 line",width=1,dpi=300, c="green")
        display(p1)
        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\FPpred_v_CNLpred$(data_mode).png")
        end
        p2 = scatter(df_mode_residuals_FP, df_mode_residuals_CNL,label=false, alpha=0.7, xlabel="Structure-based predicted logIE residuals", ylabel="MS-based predicted logIE residuals", legend=:best, c="magenta", xlims=(-2.5,2.8), ylims=(-2.4,3.4), dpi=300)
        plot!([0,0],[minimum(df_mode_residuals_CNL),maximum(df_mode_residuals_CNL)],label=false,width=1,dpi=300, c="black")
        plot!([minimum(df_mode_residuals_FP),maximum(df_mode_residuals_FP)],[0,0],label=false,width=1,dpi=300, c="black")
        display(p2)
        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\FPpred_v_CNLpred_$(data_mode)_residuals.png")
        end
    end
    return(df_mode,unique(failed_inchikeys))     
end

df_min, failed_min = create_df_mode("min", allowplots=true, allowsave=true)
df_mean, failed_mean = create_df_mode("mean", allowplots=true, allowsave=true)
df_max, failed_max = create_df_mode("max", allowplots=true, allowsave=true)

