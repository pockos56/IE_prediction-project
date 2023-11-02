using Pkg
Pkg.add(url="https://github.com/pockos56/IE_prediction.jl")

using CSV
using DataFrames
using StatsBase, Statistics
using IE_prediction
using Plots
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

for i = 31:size(result,1)
    frags_temp = eval(Meta.parse(result[i,:MatchedFrags]))
    CNL_temp = result[i,:MeasMass] .- frags_temp
    result[i,:logIE_CNL] = logIE_from_CNLs(CNL_temp, result[i,:MeasMass],"positive",2.7)     # IE_CNL prediction
    result[i,:logIE_SMILES] = logIE_from_InChIKey(String(result[i,:INCHIKEY]),"positive",2.7)     # IE_SMILES prediction
    println("Completed: $i/$(size(result,1))")
end

#Plots
y_CNL = result[:,:logIE_CNL]
y_smiles = result[:,:logIE_SMILES]
R2_pred = R2_eq(y_CNL,y_smiles)

p1 = scatter(y_smiles,y_CNL,label=false, legend=:best, title="Pred vs pred", color = :magenta, xlabel="SMILES-Predicted log(IE)", ylabel = "CNL-Predicted log(IE)",dpi=300)
plot!([minimum(y_smiles),maximum(y_smiles)],[minimum(y_smiles),maximum(y_smiles)],label="1:1 line",width=2,dpi=300)
annotate!(maximum(y_smiles),0.8+minimum(y_smiles),latexstring("R^2=$(round(R2_pred, digits=3))"),:right)
savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Graphs\\Validation\\Pred_pred.png")

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
        savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_$ESI_name.png")
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
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Regression_pHcolor_$ESI_name.png")
        end

        plot_pH_res = scatter(y_train,z6,label="Training set", legend=:best, title = "ESI$(ESI_symbol) Regression residuals",markershape=:circle, marker_z=X_train[:,1],color = :jet, xlabel = "Experimental log(IE)", ylabel = "Residual")
        scatter!(y_test,z7, markershape=:rect,marker_z=X_test[:,1], label="Test set",color=:jet)
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[0,0],label="1:1 line",width=2) # 1:1 line
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[3*std(vcat(z6,z7)),3*std(vcat(z6,z7))],label="+/- 3 std",linecolor ="grey",width=2) # +3 sigma
        plot!([minimum(vcat(y_test,y_train)),maximum(vcat(y_test,y_train))],[-3*std(vcat(z6,z7)),-3*std(vcat(z6,z7))],label=false,linecolor ="grey",width=2) #-3 sigma

        if allowsave == true
            savefig("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\Graphs\\CNL\\CNL_Cat_Residuals_pHcolor_$ESI_name.png")
        end
        display(plot_pH)
        display(plot_pH_res)
    end



