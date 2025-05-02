using CSV, DataFrames, JLD, Plots, ScikitLearn, Statistics, Conda, PyCall, BSON
import StatsPlots as sp
using ProgressBars
using Distributions

#include("function getVec(matStr).jl")
data_set = NIST
i=1
k=1

str = data_set[Indeces[k][i],:MZ_VALUES]
average_spectra = spectra

function first_filtering(data_set)
    #filtering N/A and missing values of InCIkeys and Resolution columns
    data_set = data_set[.!ismissing.(data_set[!,:INCHIKEY]),:]
    data_set = data_set[(data_set[!,:INCHIKEY].!= "N/A"),:]
    data_set = data_set[(data_set[!,:INCHIKEY].!= "NA"),:]              # Send to Vika for addition
    
    data_set = data_set[.!ismissing.(data_set[!,:MZ_VALUES]),:]
    data_set = data_set[(data_set[!,:MZ_VALUES].!= "[]"),:]
    
 

    #filtering resolution < 5000
    data_set= data_set[.!ismissing.(data_set[!,:RESOLUTION]),:]
    data_set= data_set[.!isnan.(data_set[!,:RESOLUTION]),:]                     # Send to Vika for addition (IMPORTANT)

    unique(data_set[!,:RESOLUTION])
    temp =[]
    for i = 1:length(data_set[!,:RESOLUTION])
        try
            if  typeof(data_set[i,:RESOLUTION]) == String
                if parse(Int64,data_set[i,:RESOLUTION]) < 5000
                    push!(temp,i)
                end 
            elseif typeof(data_set[i,:RESOLUTION]) == Int64
                if data_set[i,:RESOLUTION] < 5000
                    push!(temp,i)
                end  
            end


        catch
            push!(temp,i)
        end
    end

    deleteat!(data_set,temp)

    return data_set
end

function filtering_IonMode(data_set)
    #filter positive (H+) and negative modes
    negative_ind = []
    positive_ind = []
    for i in ProgressBar(1:length(data_set[!,:ION_MODE]))           # Send to Vika for addition
        if (data_set[i,:ION_MODE] == "POSITIVE" || data_set[i,:ION_MODE] == "P")# && (data_set[i,:PRECURSOR_ION] - data_set[i,:EXACT_MASS] < 1.01)
            push!(positive_ind, i)
        elseif (data_set[i,:ION_MODE] == "NEGATIVE" || data_set[i,:ION_MODE] == "N")# && (data_set[i,:EXACT_MASS] - data_set[i,:PRECURSOR_ION] < 1.01)
            push!(negative_ind, i)
        end
    end
    negative_set = data_set[negative_ind,:]
    positive_set = data_set[positive_ind,:]
    return positive_set, negative_set
end

function av_spectra(data_set)
    spectra = DataFrame()
    # find all unique chemicals and their indeces
    temp1 = unique(data_set[!,:INCHIKEY])
    TV1 = [temp1 [count(==(i),data_set[!,:INCHIKEY]) for i in temp1]]
    TV1 = TV1[TV1[:,2].!=1,:]

    Indeces = [findall(data_set[!,:INCHIKEY].== TV1[i,1]) for i= 1:size(TV1,1)]
    emptyrows = []
    for k in ProgressBar(1:length(Indeces))       
        m = []
        temp = []
        for i = 1:length(Indeces[k])
            TV2 = eval(Meta.parse(replace(data_set[Indeces[k][i],:MZ_VALUES], r"Any(\[.*?\])" => s"\1")))
            CNLs = [round(data_set[Indeces[k][i],:PRECURSOR_ION]- TV2[l],digits = 3) for l= 1:length(TV2)]
            CNLs = CNLs[CNLs .>= 1.007]
            m = [m;CNLs]
            temp = [temp; data_set[Indeces[k][i],:PRECURSOR_ION]]         
        end
        m = unique(m)
        
        if m == [] || temp == []
            push!(emptyrows,k)
        else
            append!(spectra,[DataFrame(INCHIKEY = data_set[Indeces[k][1],:INCHIKEY]) DataFrame(INDECES = [Indeces[k]]) DataFrame(CNLs = [m]) DataFrame(PRECURSOR_ION = [temp])])
        end 
    end
    return spectra
end

function CNLs_list(average_spectra; threshold::Int=80)

    #Creating a list of CNLs 
    CNLs_all = []
    for i = 1:nrow(average_spectra)
        if typeof(average_spectra[i,"CNLs"])== String
            TV = eval(Meta.parse(replace(average_spectra[i,"CNLs"], r"Any(\[.*?\])" => s"\1")))
        else
            TV = average_spectra[i,"CNLs"]
        end
        CNLs_all =[CNLs_all; (round.(TV,digits =2))]
    end
    unCNL =unique(CNLs_all)
    CNLmax = []
    P_CNL = []

    TV = [count(==(i),CNLs_all) for i in unCNL]
    push!(CNLmax,unCNL[argmax(TV)])
    push!(P_CNL,100*maximum(TV)/length(CNLs_all))
    x = TV
    while maximum(TV) > threshold
        deleteat!(unCNL,argmax(TV))
        deleteat!(TV,argmax(TV))
        push!(CNLmax,unCNL[argmax(TV)])
        push!(P_CNL,100*maximum(TV)/length(CNLs_all))
    end
    return CNLmax,P_CNL
end

NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)
positive_set, negative_set = filtering_IonMode(first_filtering(NIST))
CNLmax_neg = CNLs_list(av_spectra(negative_set),threshold=15)[1]
CNLmax_pos = CNLs_list(av_spectra(positive_set),threshold=80)[1]