using CSV, DataFrames, ProgressBars, Plots, LaTeXStrings, Statistics, StatsPlots

function create_comparison_plots(data_mode::String; allowplots::Bool=false, allowsave::Bool=false)
    df_FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_FP_$data_mode.csv", DataFrame)
    df_CNL = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\models\\y_hat_df_CNL_$data_mode.csv", DataFrame)

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
    df_mode.CNL_hat_FP_hat_residual = df_mode[:,"IE_hat_CNL"] - df_mode[:,"IE_hat_FP"]
    df_mode.df_mode_residuals_FP = df_mode[:,"IE_hat_FP"] - df_mode[:,"IE"]
    df_mode.df_mode_residuals_CNL = df_mode[:,"IE_hat_CNL"] - df_mode[:,"IE"]

    df_mode_residuals_FP = df_mode[:,"IE_hat_FP"] - df_mode[:,"IE"]
    df_mode_residuals_CNL = df_mode[:,"IE_hat_CNL"] - df_mode[:,"IE"]

    MAE = round((mean(abs.(df_mode.CNL_hat_FP_hat_residual))), digits=2)
    RMSE = round(sqrt(mean((df_mode.CNL_hat_FP_hat_residual).^2)), digits=2)
    R2 = 1 - (sum((df_mode[:,"IE_hat_CNL"] .- df_mode[:,"IE_hat_FP"]).^2) / sum((df_mode[:,"IE_hat_CNL"] .- mean(df_mode[:,"IE_hat_CNL"])).^2))
    #

    #Plots
    if allowplots
        #  Scatter plot
        lim_x_min = -0.7
        lim_x_max = 5.7
        lim_y_min = -0.8
        lim_y_max = 5.8

        p1 = scatter(df_mode[:,"IE_hat_FP"],df_mode[:,"IE_hat_CNL"],label=false, alpha=0.7, xlabel="Structure-based predicted logIE", ylabel=L"MS$^2$-based predicted logIE", legend=:best, c="magenta", xlims=(lim_x_min,lim_x_max),ylims=(lim_y_min,lim_y_max),dpi=300)
        plot!([lim_x_min, lim_x_max],[lim_x_min, lim_x_max],label="1:1 line",width=1.5,dpi=300, c="green")
        annotate!(lim_x_max - 0.4, 0.6 + lim_y_min, latexstring("R^2=$(round(R2, digits=3))"),:right)

        display(p1)
        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\FPpred_v_CNLpred$(data_mode).png")
        end

        lim_x_min = -2.5
        lim_x_max = 2.8
        lim_y_min = -2.8
        lim_y_max = 3.6

        # Scatter residual plot
        p2 = scatter(df_mode_residuals_FP, df_mode_residuals_CNL,label=false, alpha=0.7, xlabel="Structure-based predicted logIE residuals", ylabel=L"MS$^2$-based predicted logIE residuals", legend=:best, c="magenta", xlims=(lim_x_min, lim_x_max), ylims=(lim_y_min, lim_y_max), dpi=300)
        plot!([0,0],[lim_y_min, lim_y_max],label=false,width=1,dpi=300, c="black")
        plot!([lim_x_min, lim_x_max],[0,0],label=false,width=1,dpi=300, c="black")
        display(p2)
        if allowsave
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\FPpred_v_CNLpred_$(data_mode)_residuals.png")
        end
    end
    return(df_mode, unique(failed_inchikeys), MAE, RMSE)     
end

function create_boxplots(df_min, df_mean, df_max; allowsave=false)
    df_min.INCHIKEY .= "min"
    df_mean.INCHIKEY .= "mean"
    df_max.INCHIKEY .= "max"

    p4 = boxplot(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, ylabel = "logIE residual")
    boxplot!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300)
    boxplot!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300)
    display(p4)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_boxplot.png")
    end

    p5 = violin(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, ylabel = "logIE residual")
    violin!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300)
    violin!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300)
    display(p5)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_violin.png")
    end

    #=
    p6 = boxplot(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, outliers=false,alpha=0.6, c="blue2", boxcolor = 0, fillalpha = 0, width=0.1)
    boxplot!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300, outliers=false, c="orange1")
    boxplot!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300, outliers=false, c="green2")
    violin!(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, c="blue1")
    violin!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300, c="orange1")
    violin!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300, c="green2")
    display(p6)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_violin_boxplot.png")
    end
    =#
    under1_ratio_min = sum(df_min[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_min[:,"CNL_hat_FP_hat_residual"])
    under1_ratio_mean = sum(df_mean[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_mean[:,"CNL_hat_FP_hat_residual"])
    under1_ratio_max = sum(df_max[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_max[:,"CNL_hat_FP_hat_residual"])
    println("Less than 1: min: $(round(under1_ratio_min, digits=3)), mean: $under1_ratio_mean, max: $under1_ratio_max")
    min_80 = round(quantile(abs.(df_min[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)
    mean_80 = round(quantile(abs.(df_mean[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)
    max_80 = round(quantile(abs.(df_max[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)

    println("80th quantile: min: $(min_80), mean $(mean_80), max $(max_80)")
    println("80th quantile (10^): min: $(10^(min_80)), mean $(10^(mean_80)), max $(10^(max_80))")
    
    println("MAE: min: $MAE_min, mean: $MAE_mean, max: $MAE_max")
    println("RMSE: min: $RMSE_min, mean: $RMSE_mean, max: $RMSE_max")
end

df_min, failed_min, MAE_min, RMSE_min = create_comparison_plots("min", allowplots=true, allowsave=true)
df_mean, failed_mean, MAE_mean, RMSE_mean = create_comparison_plots("mean", allowplots=true, allowsave=true)
df_max, failed_max, MAE_max, RMSE_max = create_comparison_plots("max", allowplots=true, allowsave=true)
create_boxplots(df_min, df_mean, df_max, allowsave=true)

# Training and test subset => different boxplots.

function create_boxplots_train_test(df_min, df_mean, df_max; allowsave=false)
    df_min_train = df_min[df_min.class_CNL .== "train",:]
    df_mean_train = df_mean[df_mean.class_CNL .== "train",:]
    df_max_train = df_max[df_max.class_CNL .== "train",:]
    df_min_test = df_min[df_min.class_CNL .== "test",:]
    df_mean_test = df_mean[df_mean.class_CNL .== "test",:]
    df_max_test = df_max[df_max.class_CNL .== "test",:]

    df_min_train.INCHIKEY .= "min"
    df_mean_train.INCHIKEY .= "mean"
    df_max_train.INCHIKEY .= "max"
    df_min_test.INCHIKEY .= "min"
    df_mean_test.INCHIKEY .= "mean"
    df_max_test.INCHIKEY .= "max"

    p4 = boxplot(df_min_train.INCHIKEY, df_min_train[:,"CNL_hat_FP_hat_residual"], legend=false,alpha=0.7, dpi=300, ylabel = "logIE residual")
    boxplot!(df_mean_train.INCHIKEY, df_mean_train[:,"CNL_hat_FP_hat_residual"], alpha=0.7, dpi=300)
    boxplot!(df_max_train.INCHIKEY, df_max_train[:,"CNL_hat_FP_hat_residual"],alpha=0.7, dpi=300)
    boxplot!(df_min_test.INCHIKEY, df_min_test[:,"CNL_hat_FP_hat_residual"],alpha=0.7, dpi=300)
    boxplot!(df_mean_test.INCHIKEY, df_mean_test[:,"CNL_hat_FP_hat_residual"],alpha=0.7, dpi=300)
    boxplot!(df_max_test.INCHIKEY, df_max_test[:,"CNL_hat_FP_hat_residual"],alpha=0.7, dpi=300)

    display(p4)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_boxplot_train-test.png")
    end

    p5 = violin(df_min_train.INCHIKEY, df_min_train[:,"CNL_hat_FP_hat_residual"], legend=false, ylabel = "logIE residual", dpi=300, c=:yellow3, side=:left, xtickfont=16)
    violin!(df_min_test.INCHIKEY, df_min_test[:,"CNL_hat_FP_hat_residual"], dpi=300, alpha=0.6, c=:yellow3, side=:right)
    violin!(df_mean_train.INCHIKEY, df_mean_train[:,"CNL_hat_FP_hat_residual"], dpi=300, c=:pink2, side=:left)
    violin!(df_mean_test.INCHIKEY, df_mean_test[:,"CNL_hat_FP_hat_residual"], dpi=300,alpha=0.6, c=:pink2, side=:right)
    violin!(df_max_train.INCHIKEY, df_max_train[:,"CNL_hat_FP_hat_residual"], dpi=300, c=:lightblue2,side=:left)
    violin!(df_max_test.INCHIKEY, df_max_test[:,"CNL_hat_FP_hat_residual"], dpi=300,alpha=0.6, c=:lightblue2, side=:right)
    display(p5)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_violin_train-test.png")
    end

    #=
    p6 = boxplot(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, outliers=false,alpha=0.6, c="blue2", boxcolor = 0, fillalpha = 0, width=0.1)
    boxplot!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300, outliers=false, c="orange1")
    boxplot!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300, outliers=false, c="green2")
    violin!(df_min.INCHIKEY, df_min[:,"CNL_hat_FP_hat_residual"], legend=false, dpi=300, c="blue1")
    violin!(df_mean.INCHIKEY, df_mean[:,"CNL_hat_FP_hat_residual"], dpi=300, c="orange1")
    violin!(df_max.INCHIKEY, df_max[:,"CNL_hat_FP_hat_residual"], dpi=300, c="green2")
    display(p6)
    if allowsave
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\Graphs\\CNL vs FP\\Residual_violin_boxplot.png")
    end
    =#
    under1_ratio_min = sum(df_min[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_min[:,"CNL_hat_FP_hat_residual"])
    under1_ratio_mean = sum(df_mean[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_mean[:,"CNL_hat_FP_hat_residual"])
    under1_ratio_max = sum(df_max[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_max[:,"CNL_hat_FP_hat_residual"])
    println("Less than 1: min: $(round(under1_ratio_min, digits=3)), mean: $under1_ratio_mean, max: $under1_ratio_max")
    min_80 = round(quantile(abs.(df_min[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)
    mean_80 = round(quantile(abs.(df_mean[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)
    max_80 = round(quantile(abs.(df_max[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)

    println("80th quantile: min: $(min_80), mean $(mean_80), max $(max_80)")
    println("80th quantile (10^): min: $(10^(min_80)), mean $(10^(mean_80)), max $(10^(max_80))")
    
    println("MAE: min: $MAE_min, mean: $MAE_mean, max: $MAE_max")
    println("RMSE: min: $RMSE_min, mean: $RMSE_mean, max: $RMSE_max")
end


#=
df_min_train, df_min_test, failed_min, MAE_min, RMSE_min = create_comparison_plots_test_only("min", allowplots=true, allowsave=true)
df_mean_train, df_mean_test, failed_mean, MAE_mean, RMSE_mean = create_comparison_plots_test_only("mean", allowplots=true, allowsave=true)
df_max_train, df_mean_test, failed_max, MAE_max, RMSE_max = create_comparison_plots_test_only("max", allowplots=true, allowsave=true)
create_boxplots(df_min, df_mean, df_max, allowsave=true)
=#
