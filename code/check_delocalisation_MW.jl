########################################################################
### Goal: Check if delocalisation and molecular weight is correlated with the prediction error
using Plots, Statistics, DataFrames, CSV, PyCall, Conda, ProgressBars
pcp = pyimport("pubchempy")
alc = pyimport("rdkit.Chem.AllChem")
dr = pyimport("rdkit.Chem.Draw")

## Paths
project_path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\"
path_data = joinpath(project_path, "data")
path_graphs = joinpath(project_path, "Graphs")

## Load files
FP = CSV.read(joinpath(path_data,"Fingerprints","FP6_mean.csv"), DataFrame)
validation_inchikeys = CSV.read(joinpath(path_data, "Validation_set_inchikeys.csv"), DataFrame)
deleteat!(FP, findall(x -> x in validation_inchikeys.INCHIKEY, FP.INCHIKEY))

## Calculate resonance structures, as a good indicator of delocalisation
resonance_structures_no = Int.(zeros(size(FP,1)))
for i in ProgressBar(1:size(FP,1))
    smiles = FP[i,:SMILES]
    resonance_structures_no[i] = Int(length(alc.ResonanceMolSupplier(alc.MolFromSmiles(smiles), alc.ResonanceFlags.UNCONSTRAINED_ANIONS)))
end

## Find the prediction error for the FP model
# Retrieving the y_hat_df_mean variable from FP_regression.jl file
y_hat_df_mean[:,:Resonance_No] .= resonance_structures_no
y_hat_df_mean[:,:Residuals] = y_hat_df_mean[:,:IE_hat_fp] - y_hat_df_mean[:,:IE]
y_hat_df_mean[:,:Residuals_abs] = abs.(y_hat_df_mean.Residuals)
y_hat_df_mean[:,:Relative_error] = (y_hat_df_mean.Residuals_abs) ./ y_hat_df_mean[:,:IE]
    
df_train = y_hat_df_mean[y_hat_df_mean[:,:class_fp].=="train",:]
df_test = y_hat_df_mean[y_hat_df_mean[:,:class_fp].=="test",:]

# Mean residuals (clean)
    # Group together to de-noise data (True)
    df_train_mean_res = combine(groupby(df_train, :Resonance_No), :Residuals => mean => :AverageResiduals)
    df_test_mean_res = combine(groupby(df_test, :Resonance_No), :Residuals => mean => :AverageResiduals)
    # To see if there is an overestimation
    cor_mean_train = cor(df_train_mean_res[:,:Resonance_No], df_train_mean_res[:,:AverageResiduals])  # -0.15   
    cor_mean_test = cor(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals])     # 0.38
    # To see if there is an overestimation (Plot)
    p1 = scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals],xlims=(0,40))
    p2 = scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals],xlims=(0,110))
    p3 = scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals])
    p4 = scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", legend=false)
    # To see if the errors are bigger
    cor_mean_train = cor(df_train_mean_res[:,:Resonance_No], abs.(df_train_mean_res[:,:AverageResiduals]))  # 0.08   
    cor_mean_test = cor(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]))     # 0.54
    # To see if the errors are bigger (Plot)
    p1 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]),xlims=(0,40))
    p2 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]),xlims=(0,110))
    p3 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]))
    p4 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]), xscale=:log10, xlabel="Number of resonance structures", ylabel="Absolute prediction error", legend=false,dpi=300)

    # Check correlation for raw data (False)
    cor_test = cor(df_test[:,:Resonance_No], df_test[:,:Residuals])     # 0.11
    cor_abs_test = cor(df_test[:,:Resonance_No], df_test[:,:Residuals_abs])     # 0.13
    scatter(df_test[:,:Resonance_No], abs.(df_test[:,:Residuals]), xscale=:log10, xlabel="Number of resonance structures", ylabel="Absolute prediction error", label="All", alpha=0.7)
    scatter(df_test[:,:Resonance_No], df_test[:,:Residuals], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", label="All data points")
    savefig(joinpath(path_graphs,"Resonance", "Raw_absolute_error.pdf"))

    # Group together to de-noise data - Residuals_abs (True)
    df_train_mean_res = combine(groupby(df_train, :Resonance_No), :Residuals_abs => mean => :Residuals_abs)
    df_test_mean_res = combine(groupby(df_test, :Resonance_No), :Residuals_abs => mean => :Residuals_abs)
    # To see if there is an overestimation
    cor_mean_train = cor(df_train_mean_res[:,:Resonance_No], df_train_mean_res[:,:Residuals_abs])  # -0.38   
    cor_mean_test = cor(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:Residuals_abs])     # 0.50

    scatter!(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:Residuals_abs]), xscale=:log10, xlabel="Number of resonance structures", ylabel="Absolute prediction error", label="Grouped", dpi=300, alpha=0.8)
    savefig(joinpath(path_graphs,"Resonance", "absolute_error_joined.pdf"))
    # Relative_error (No)
    cor_mean_train = cor(df_train[:,:Resonance_No], df_train[:,:Relative_error])  # NaN  
    cor_mean_test = cor(df_test[:,:Resonance_No], df_test[:,:Relative_error])     # 0.00

    # Group together to de-noise data - Relative_error (Meh)
    df_train_mean_res = combine(groupby(df_train, :Resonance_No), :Relative_error => mean => :Relative_error)
    df_test_mean_res = combine(groupby(df_test, :Resonance_No), :Relative_error => mean => :Relative_error)

    cor_mean_train = cor(df_train_mean_res[:,:Resonance_No], df_train_mean_res[:,:Relative_error])  # NaN  
    cor_mean_test = cor(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:Relative_error])     # 0.26

    scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:Relative_error], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", legend=false)

# Median and mean absolute residuals (clean)
    # Group together to de-noise data
    df_train_mean_abs = combine(groupby(df_train, :Resonance_No), :Residuals_abs => mean => :AverageResiduals)
    df_test_mean_abs = combine(groupby(df_test, :Resonance_No), :Residuals_abs => mean => :AverageResiduals)
    # To see if there is an overestimation
    cor_mean_train = cor(df_train_mean_abs[:,:Resonance_No], df_train_mean_abs[:,:AverageResiduals])  # -0.38  
    cor_mean_test = cor(df_test_mean_abs[:,:Resonance_No], df_test_mean_abs[:,:AverageResiduals])     # 0.50


    # To see if there is an overestimation (Plot)
    p1 = scatter(df_test_mean_abs[:,:Resonance_No], df_test_mean_abs[:,:AverageResiduals],xlims=(0,40))
    p2 = scatter(df_test_mean_abs[:,:Resonance_No], df_test_mean_abs[:,:AverageResiduals],xlims=(0,110))
    p3 = scatter(df_test_mean_abs[:,:Resonance_No], df_test_mean_abs[:,:AverageResiduals])
    p4 = scatter(df_test_mean_abs[:,:Resonance_No], df_test_mean_abs[:,:AverageResiduals], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", legend=false)
    # To see if the errors are bigger
    cor_mean_train = cor(df_train_mean_res[:,:Resonance_No], abs.(df_train_mean_res[:,:AverageResiduals]))  # 0.08   
    cor_mean_test = cor(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]))     # 0.54
    cor_median_train = cor(df_train_median_res[:,:Resonance_No], abs.(df_train_median_res[:,:MedianResiduals]))  # 0.05 
    cor_median_test = cor(df_test_median_res[:,:Resonance_No], abs.(df_test_median_res[:,:MedianResiduals]))     # 0.53
    # To see if the errors are bigger (Plot)
    p1 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]),xlims=(0,40))
    p2 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]),xlims=(0,110))
    p3 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]))
    p4 = scatter(df_test_mean_res[:,:Resonance_No], abs.(df_test_mean_res[:,:AverageResiduals]), xscale=:log10, xlabel="Number of resonance structures", ylabel="Absolute prediction error", legend=false)

    # Check correlation for raw data
    cor_test = cor(df_test[:,:Resonance_No], df_test[:,:Residuals])     # 0.11
    cor_abs_test = cor(df_test[:,:Resonance_No], abs.(df_test[:,:Residuals]))     # 0.13
    scatter(df_test[:,:Resonance_No], abs.(df_test[:,:Residuals]), xscale=:log10, xlabel="Number of resonance structures", ylabel="Absolute prediction error", legend=false)
    scatter(df_test[:,:Resonance_No], df_test[:,:Residuals], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", legend=false)



# Relative Standard Error (clean)
    df = df_train
    function ReRse(df::DataFrame)
        predicted_vals = df[:,:IE_hat_fp]
        true_vals = df[:,:IE]
        df.abs_relative_error = (0.00000001.+abs.(predicted_vals .- true_vals)) ./ abs.(true_vals)
        df.relative_error = (0.00000001.+(predicted_vals .- true_vals)) ./ abs.(true_vals)
        std_error = std(predicted_vals .- true_vals) / sqrt(length(predicted_vals .- true_vals))
        RSE = (std_error / mean(predicted_vals)) * 100
    end
    ReRse(df_train)
    ReRse(df_test)
    histogram(df_test.abs_relative_error)
    histogram(df_test.relative_error)
    std(predicted_vals .- true_vals)
    std(abs.(predicted_vals .- true_vals))
    cor_abs_test = cor(df_test[:,:Resonance_No], df_test[:,:abs_relative_error])     # -0.02
    cor_test = cor(df_test[:,:Resonance_No], df_test[:,:relative_error])     # -0.02
    # To see if there is an overestimation (Plot)
    p1 = scatter(df_test[:,:Resonance_No], df_test[:,:abs_relative_error],ylims=(0,10),xscale=:log10)
    p2 = scatter(df_test[:,:Resonance_No], df_test[:,:relative_error],xscale=:log10,ylims=(-10,10))
    p3 = scatter(df_test_mean_res[:,:Resonance_No], df_test_mean_res[:,:AverageResiduals], xscale=:log10, xlabel="Number of resonance structures", ylabel="Prediction error", legend=false)

    #df_train_res = combine(groupby(df_train, :Resonance_No), :relative_error => mean => :AverageRelError)
    df_test_res = combine(groupby(df_test, :Resonance_No), :relative_error => mean => :AverageRelError)
    #df_train_abs_res = combine(groupby(df_train, :Resonance_No), :abs_relative_error => mean => :AverageRelError)
    df_test_abs_res = combine(groupby(df_test, :Resonance_No), :abs_relative_error => mean => :AverageRelError)

    cor_abs_test = cor(df_test_abs_res[:,:Resonance_No], df_test_abs_res[:,:AverageRelError])     # 0.04
    cor_test = cor(df_test_res[:,:Resonance_No], df_test_res[:,:AverageRelError])     # 0.11

# MW correlation
    df_test.MW .= 0.0
    for i in ProgressBar(1:size(df_test,1))
        try
            df_test.MW[i] = parse(Float64, pcp.get_compounds(df_test[i,"INCHIKEY"], "inchikey")[1].molecular_weight)
        catch
            continue
        end
    end
    scatter(df_test.Residuals, df_test.MW, ylabel="Residual (in log units)", xlabel="Molecular weight")

    # All test compounds
    cor_mean_test = cor(df_test[:,:MW], df_test[:,:Residuals_abs])     # r = 0.04
    p4 = scatter(df_test[:,:MW], (df_test[:,:Residuals_abs]), xlabel="Molecular weight", ylabel="Absolute prediction error", label="All",dpi=300, alpha=0.8)

    # Group together
    df_test_mean_res = combine(groupby(df_test, :MW), :Residuals_abs => mean => :Residuals_abs)
    cor_mean_test = cor(df_test_mean_res[:,:MW], df_test_mean_res[:,:Residuals_abs])     # r = 0.03

    scatter!(df_test_mean_res[:,:MW], abs.(df_test_mean_res[:,:Residuals_abs]), xlabel="Molecular weight", ylabel="Absolute prediction error", label="Grouped", dpi=300, alpha=0.8)
    savefig(joinpath(path_graphs,"MW", "absolute_error_joined.pdf"))

    MW_Residual_correlation = cor(y_hat_df_mean.MW, y_hat_df_mean.Residuals)       
    MW_Residual_correlation_train = cor(y_hat_df_mean[y_hat_df_mean[:,"class_fp"].=="train",:MW], abs.(y_hat_df_mean[y_hat_df_mean[:,"class_fp"].=="train",:Residuals]))       
    MW_Residual_correlation_test = cor(y_hat_df_mean[y_hat_df_mean[:,"class_fp"].=="test",:MW], abs.(y_hat_df_mean[y_hat_df_mean[:,"class_fp"].=="test",:Residuals]))
    println("Is MW correlated with the prediction error? \nr=$(MW_Residual_correlation) \nr_train=$(MW_Residual_correlation_train) \nr_test=$(MW_Residual_correlation_test)\n")       

    y_hat_df_mean_backup
    y_hat_df_mean = y_hat_df_mean[y_hat_df_mean[:,"Resonance_No"].<30,:]    # 95% of the compounds have fewer than 30 resonance structures


