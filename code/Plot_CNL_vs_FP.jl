########################################################################
## Goal: Evaluate the CNL predictions using the FP predictions as baseline

using CSV, DataFrames, ProgressBars, Plots, LaTeXStrings, Statistics, StatsPlots

## Paths
project_path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\"
path_data = joinpath(project_path, "data")
path_graphs = joinpath(project_path, "Graphs")

# Comparison plots
function create_comparison_plots(data_mode::String; allowplots::Bool=false, allowsave::Bool=false)
    df_FP = CSV.read(joinpath(path_data, "models", "y_hat_df_FP_$data_mode.csv"), DataFrame)
    df_CNL = CSV.read(joinpath(path_data, "models", "y_hat_df_CNL_$data_mode.csv"), DataFrame)

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

    # Plots if allowplots is set to true
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
            savefig(joinpath(path_graphs, "CNL vs FP", "FPpred_v_CNLpred$(data_mode).pdf"))
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
            savefig(joinpath(path_graphs, "CNL vs FP", "FPpred_v_CNLpred_$(data_mode)_residuals.pdf"))
        end
    end
    return(df_mode, unique(failed_inchikeys), MAE, RMSE)     
end

# Function to plot training and test subset
function create_boxplots_train_test_only_mean(df_mean; allowsave=false)
    df_mean_train = df_mean[df_mean.class_CNL .== "train",:]
    df_mean_test = df_mean[df_mean.class_CNL .== "test",:]
    df_mean_train.INCHIKEY .= "mean"
    df_mean_test.INCHIKEY .= "mean"

    p4 = boxplot(df_mean_train.INCHIKEY, df_mean_train[:,"CNL_hat_FP_hat_residual"], legend=false,alpha=0.7, dpi=300, ylabel = "logIE residual")
    boxplot!(df_mean_test.INCHIKEY, df_mean_test[:,"CNL_hat_FP_hat_residual"],alpha=0.7, dpi=300)

    display(p4)
    if allowsave
        savefig(joinpath(path_graphs, "CNL vs FP", "Residual_boxplot_train-test.pdf"))
    end

    p5 = violin(df_mean_train.INCHIKEY, df_mean_train[:,"CNL_hat_FP_hat_residual"], ylabel = "logIE residual", dpi=300, alpha=0.6, c=:yellow3, side=:left, xtickfont=16, xaxis=false, legend=false, size=(400,400))
    violin!(df_mean_test.INCHIKEY, df_mean_test[:,"CNL_hat_FP_hat_residual"], dpi=300,alpha=0.6, c=:pink2, side=:right)
        
    display(p5)
    if allowsave
        savefig(joinpath(path_graphs, "CNL vs FP", "Residual_violin_train-test_mean_only.pdf"))
    end

    under1_ratio_mean = sum(df_mean[:,"CNL_hat_FP_hat_residual"] .<= 1)/length(df_mean[:,"CNL_hat_FP_hat_residual"])
    println("Less than 1: min: $(round(under1_ratio_min, digits=3)), mean: $under1_ratio_mean, max: $under1_ratio_max")
    mean_80 = round(quantile(abs.(df_mean[:,"CNL_hat_FP_hat_residual"]),0.80), digits=3)

    println("80th quantile: min: $(min_80), mean $(mean_80), max $(max_80)")
    println("80th quantile (10^): min: $(10^(min_80)), mean $(10^(mean_80)), max $(10^(max_80))")
    
    println("MAE: min: $MAE_min, mean: $MAE_mean, max: $MAE_max")
    println("RMSE: min: $RMSE_min, mean: $RMSE_mean, max: $RMSE_max")
end

