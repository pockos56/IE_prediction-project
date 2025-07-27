########################################################################
### Goal: Check if the compounds with the highest prediction errors

using Plots, Statistics, DataFrames, CSV, PyCall, Conda, ProgressBars, StatsPlots
pcp = pyimport("pubchempy")
alc = pyimport("rdkit.Chem.AllChem")

## Paths
project_path = "C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\"
path_data = joinpath(project_path, "data")
path_graphs = joinpath(project_path, "Graphs")

# Function that associates INCHIKEY to name
function inchikey_to_names(inchikey_list::Vector)
    FP = CSV.read(joinpath(path_data,"Fingerprints","FP6_mean.csv"), DataFrame)
    names_1 = []
    names_2 = []
    for i in ProgressBar(inchikey_list)
        push!(names_1, FP[FP[:,:INCHIKEY] .== i,:name])
        try
            push!(names_2, pcp.get_compounds(i, "inchikey")[1].synonyms[1])
        catch
            continue
        end
    end
    return names_1, names_2
end

results_fp = deepcopy(y_hat_df_mean)
results_fp.diff = results_fp.IE_hat_fp - results_fp.IE
results_fp.diff_abs = abs.(results_fp.diff)
results_fp = sort(results_fp, :diff_abs, rev=true)[1:10,:]
fp_1, fp_2 = inchikey_to_names(results_fp[:,:INCHIKEY])
results_cnl = deepcopy(y_hat_df_CNL)
cnl_1, cnl_2 = inchikey_to_names(results_cnl[:,:INCHIKEY])
results_fp.name = fp_2
results_cnl.name = cnl_2

# Check intersection of two test sets
unique_comps_fp = unique(results_fp[:,:INCHIKEY])   # 1144
unique_comps_cnl = unique(results_cnl[:,:INCHIKEY]) # 768
intersect(unique_comps_cnl, unique_comps_fp)        # 764
unique_comps_fp_test = unique(results_fp[results_fp[:,:class_fp].=="test",:INCHIKEY])   # 229
unique_comps_cnl_test = unique(results_cnl[results_cnl[:,:class_CNL].=="test",:INCHIKEY])   # 154
intersect(unique_comps_cnl_test, unique_comps_fp_test)  # 31

# Plots
scatter(results_cnl[1:10,:resonance])
scatter!(results_fp[:,:resonance])
boxplot(results_cnl[1:10,:MW])
boxplot!(results_fp[:,:MW])

# Check if the FP and CNL model have overlapping test compounds.
results_fp = deepcopy(y_hat_df_mean)
results_cnl = deepcopy(y_hat_df_CNL)

fp_test = results_fp[results_fp[:,:class_fp] .== "test",:]
cnl_test = results_cnl[results_cnl[:,:class_CNL] .== "test",:]

inchikeys_fp_test = unique(fp_test[:,:INCHIKEY])
inchikeys_cnl_test = unique(cnl_test[:,:INCHIKEY])
common_inchikeys = intersect(inchikeys_cnl_test, inchikeys_fp_test)

fp_test[findall(x->x in common_inchikeys, fp_test[:,:INCHIKEY]), :]
cnl_test[findall(x->x in common_inchikeys, cnl_test[:,:INCHIKEY]), :]

for i = 1:length(common_inchikeys)
    common_inchikey = common_inchikeys[i]
    fp_temp = fp_test[findall(x->x .== common_inchikey, fp_test[:,:INCHIKEY]), :]
    cnl_temp = cnl_test[findall(x->x .== common_inchikey, cnl_test[:,:INCHIKEY]), :]
    combine(groupby(cnl_temp, :pH_aq), "IE_hat_CNL"=>mean=> "IE_hat_CNL_average")
end
joinpath(path_data,"Fingerprints","FP6_mean.csv")
# Check if there are a lot of nitro groups =>> Yes indeed there are!
FP = CSV.read(joinpath(path_data,"Fingerprints","FP6_mean.csv"), DataFrame)
unique(FP[findall(x->x.==1, FP[:,:PubchemFP558]),:INCHIKEY])

