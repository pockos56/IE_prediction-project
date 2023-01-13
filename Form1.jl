import Pkg
Pkg.add("CSV")

# import packages
import ScikitLearn
using BSON
import LinearAlgebra
import Statistics
using DataFrames
using CSV

# load files
data1 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI-.csv", DataFrame)
data2 = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\MOESM4_ESM_ESI+.csv", DataFrame)

data_minus = data1[!, [2,3,26]]
data_plus = data2[!, [2,3,26]]

data_minus_filt = unique(data_minus,2)
data_plus_filt = unique(data_plus,2)











