########################################################################
## Goal: Perform PCA for the FP dataset compounds

using Plots, Statistics, DataFrames, CSV, PyCall, Conda, ProgressBars, ScikitLearn
pcp = pyimport("pubchempy")

# Load data
FP = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Fingerprints\\FP6_mean.csv", DataFrame)
validation_inchikeys = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Validation_set_inchikeys.csv", DataFrame)
deleteat!(FP, findall(x -> x in validation_inchikeys.INCHIKEY, FP.INCHIKEY))
groupby(FP, "inchi")

# Retrieve logP and MW using pubchempy
FP.MW .= 0.0
FP.logP .= 0.0
for i in ProgressBar(1:size(FP,1))
    try
        mol_i = pcp.get_compounds(FP[i,"INCHIKEY"], "inchikey")[1]
        FP.MW[i] = parse(Float64, mol_i.monoisotopic_mass)
        FP.logP[i] = mol_i.xlogp
    catch
        println("error for $i")
        continue
    end
end


# Function for calculation of 6 elemental mass defects (CO, CCl, CN, CS, CF, and CH)
const m_ru = [27.9949,46.9689,26.003,43.972,30.9984,13.0078] # CO, CCl, CN, CS, CF, and CH
function EMD_calc(mz_values,m_ru)
    ER = Array{Any}(undef,length(mz_values))
    for i = 1:length(mz_values)
        KM = mz_values[1].*(round.(m_ru)./m_ru) # Kendrick mass
        ER[1] = round.(round.(KM) .- KM ; digits=3)
    end
    return ER
end

FP.emd_CO .= 0.0
FP.emd_CCl .= 0.0
FP.emd_CN .= 0.0
FP.emd_CS .= 0.0
FP.emd_CF .= 0.0
FP.emd_CH .= 0.0

for i = 1:size(FP,1)
    EMDs_i = EMD_calc(FP.MW[i], m_ru)[1]
    FP.emd_CO[i] = EMDs_i[1]
    FP.emd_CCl[i] = EMDs_i[2]
    FP.emd_CN[i] = EMDs_i[3]
    FP.emd_CS[i] = EMDs_i[4]
    FP.emd_CF[i] = EMDs_i[5]
    FP.emd_CH[i] = EMDs_i[6]
end
FP[:,end-7:end]

# PCA
using ScikitLearn
@sk_import decomposition: PCA

# Load PCA variables
X = Matrix(FP[:,end-7:end])

# Mean center and scale
X = X .- mean(X,dims = 1)
X = X ./ std(X,dims=1)

# Setup PCA model
pca = PCA(n_components = size(X,2))			#define parameters of PCA model 
pca.fit(X)	

# Calculate the cumulative explained variance
pca.explained_variance_ratio_
scatter(cumsum(pca.explained_variance_ratio_).*100, size = (1280,720), dpi = 300,left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, right_margin = 5Plots.mm, legend = :topleft, xtickfont=font(13), ytickfont=font(13), guidefont=font(20), legendfont=font(13), markersize = 5)
plot!(cumsum(pca.explained_variance_ratio_).*100, size = (1280,720), legend = false, grid = false, xlabel = "Number of PCs",
        ylabel = "Explained Variance (%)", left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, right_margin = 5Plots.mm,
        xlims = (1,8), dpi = 300)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\PCA\\Explained_variance.pdf")

#Select an appropiate number of principle components. In our case 3:
pca = PCA(n_components = 3)
pca.fit(X)

# Loadings
loadings = pca.components_		#loadings of the model
PCs = ["PC1", "PC2", "PC3"]
var_names = ["MolecularWeight", "XlogP3", "CO","CCl","CN","CS","CF","CH"]
legend_order = ["MolecularWeight", "XlogP3", "CO", "CN", "CF", "CS", "CCl", "CH"]

# Plot the loadings of each variable for each PC in a grouped bar plot:
b1 = bar(loadings[1,:], group = var_names, palette = :tab10, size = (1000,500),
left_margin = 7Plots.mm, bottom_margin = 7.5Plots.mm, right_margin = 5Plots.mm,
ylabel = "Loading", legendfont = font(7), guidefont = 15, xticks = false, title = "PC 1", legend = :best, dpi = 300, ylims = (-0.5,0.7))
b2 = bar(loadings[2,:], group = var_names, palette = :tab10, size = (1000,500),
left_margin = 7Plots.mm, bottom_margin = 7.5Plots.mm, right_margin = 5Plots.mm,
 legendfont = font(11), guidefont = 15, xticks = false, title = "PC 2", legend = false , dpi = 300,ylims = (-0.5,0.7))
b3 = bar(loadings[3,:], group = var_names, palette = :tab10, size = (1000,500),
left_margin = 7Plots.mm, bottom_margin = 7.5Plots.mm, right_margin = 5Plots.mm,
 legendfont = font(5), guidefont = 15, xticks = false, title = "PC 3" , dpi = 300,ylims = (-0.5,0.7),legend_order = legend_order, legend = :false)
plot(b1,b2,b3, layout = (1,3), dpi = 300)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\PCA\\loadings.pdf")

# Calculate the scores for each compound in each PC
scores = pca.fit_transform(X)

# Plot scores
scatter(scores[:,1], scores[:,2], scores[:,3], size = (1280, 720), markerstrokewidth = 0.75, alpha = 0.75, left_margin = 7Plots.mm, bottom_margin = 5Plots.mm, right_margin = 5Plots.mm, xtickfont=font(10), ytickfont=font(10), ztickfont=font(10),guidefont=font(12), xlabel = "PC 1($(round(pca.explained_variance_ratio_[1].*100, digits=1)))%", ylabel = "PC 2($(round(pca.explained_variance_ratio_[2].*100, digits=2)))%", zlabel = "PC3($(round(pca.explained_variance_ratio_[3].*100, digits=1)))%", legendfont=font(15), zcolor=FP.unified_IEs, legend=false, colorbar = true, cgrad = :default, dpi=300)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\PCA\\scores.pdf")