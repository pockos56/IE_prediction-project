using Pkg
Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")


using CSV
using DataFrames
using StatsBase, Statistics
using IE_prediction
using Plots, ProgressBars
using LaTeXStrings

function R2_eq(y_hat_,y_)
    u = sum((y_ - y_hat_) .^2)
    v = sum((y_ .- mean(y_)) .^ 2)
    r2_result = 1 - (u/v)
    return r2_result
end

files = readdir("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Pred pred validation\\Pest and Neo mix", join=true)
data = DataFrame()
for i = 1:9
    tripl1 = CSV.read(files[i], DataFrame)
    tripl2 = CSV.read(files[i+9], DataFrame)
    tripl3 = CSV.read(files[i+18], DataFrame)
    tripl_comb = vcat(vcat(tripl1,tripl2),tripl3)
    common_inchis = intersect(tripl1[:,:INCHIKEY],tripl2[:,:INCHIKEY],tripl3[:,:INCHIKEY])
    for inchi in common_inchis
        all_spec_per_inchikey = tripl_comb[tripl_comb[:,:INCHIKEY] .== inchi,:]
        highest_factor_index = argmax(all_spec_per_inchikey[:,:MatchFactor])
        data = append!(data,DataFrame(all_spec_per_inchikey[highest_factor_index,:]))
        data[!,:Formula] = convert.(String,data[:,:Formula])
    end
end

raw_data = data[data[:,:INCHIKEY] .!= "NA",:]    # Filtering
raw_data.INCHIKEY[raw_data.INCHIKEY .== "HSMVPDGQOIQYSR-KGENOOAVSA-N"] .= "HSMVPDGQOIQYSR-UHFFFAOYSA-N"
raw_data.INCHIKEY[raw_data.INCHIKEY .== "WCXDHFDTOYPNIE-RIYZIHGNSA-N"] .= "WCXDHFDTOYPNIE-UHFFFAOYSA-N"
raw_data.INCHIKEY[raw_data.INCHIKEY .== "HOKKPVIRMVDYPB-UVTDQMKNSA-N"] .= "HOKKPVIRMVDYPB-UHFFFAOYSA-N"
raw_data.INCHIKEY[raw_data.INCHIKEY .== "CLSVJBIHYWPGQY-GGYDESQDNA-N"] .= "CLSVJBIHYWPGQY-UHFFFAOYSA-N"
raw_data.INCHIKEY[raw_data.INCHIKEY .== "CLSVJBIHYWPGQY-GGYDESQDNA-N"] .= "CLSVJBIHYWPGQY-UHFFFAOYSA-N"
result = deepcopy(raw_data)
result.logIE_CNL .= 0.00                         # Adding pred IE_CNL column
result.logIE_SMILES .= 0.00                      # Adding pred IE_SMILES column

output = DataFrame()
for i in ProgressBar(1:size(result,1)) 
    frags_temp = eval(Meta.parse(result[i,:MatchedFrags]))
    if length(unique(round.(frags_temp,digits=2))) >= 3
        result[i,:logIE_CNL] = logIE_from_CNLs(frags_temp, result[i,:MeasMass],"positive",2.7)     # IE_CNL prediction
        result[i,:logIE_SMILES] = logIE_from_InChIKey(String(result[i,:INCHIKEY]),"positive",2.7)     # IE_SMILES prediction
        output = append!(output, DataFrame(result[i,:]))
    end
end

#Plots
y_CNL = output[:,:logIE_CNL]
y_smiles = output[:,:logIE_SMILES]
R2_pred = R2_eq(y_CNL,y_smiles)

p1 = scatter(y_smiles,y_CNL,label=false, legend=:best, title="Pred vs pred", color = :magenta, xlabel="SMILES-Predicted log(IE)", ylabel = "CNL-Predicted log(IE)",dpi=300)
plot!([minimum(y_smiles),maximum(y_smiles)],[minimum(y_smiles),maximum(y_smiles)],label="1:1 line",width=2,dpi=300)
annotate!(maximum(y_smiles),0.2+minimum(y_smiles),latexstring("R^2=$(round(R2_pred, digits=3))"),:right)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Validation\\Pred_pred.png")

histogram(y_smiles,bins=39,alpha=0.80)
histogram!(y_CNL,bins=39,alpha=0.80)


