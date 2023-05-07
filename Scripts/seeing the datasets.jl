using CSV, DataFrames


M2_pos = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI+.csv", DataFrame)
M4_pos = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI+.csv", DataFrame)

M2_neg = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM2_ESM_ESI-.csv", DataFrame)
M4_neg = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\MOESM4_ESM_ESI-.csv", DataFrame)

source_m2_pos = unique(M2_pos[:,:source])
source_m2_neg = unique(M2_neg[:,:source])

source_m4_pos = unique(M4_pos[:,:source])
source_m4_neg = unique(M4_neg[:,:source])

database = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)

unique(database[:,:ION_MODE])
database_pos = vcat(database[database[:,:ION_MODE] .== "P",:], database[database[:,:ION_MODE] .== "POSITIVE",:])
database_neg = vcat(database[database[:,:ION_MODE] .== "N",:], database[database[:,:ION_MODE] .== "NEGATIVE",:])

INSTRUMENT_TYPE_pos = unique(database_pos[:,:INSTRUMENT_TYPE])
INSTRUMENT_TYPE_neg = unique(database_neg[:,:INSTRUMENT_TYPE])

MS_TYPE_pos = unique(database_pos[:,:MS_TYPE])          #0.03
database_pos[ismissing.(database_pos[:,:MS_TYPE]),:MS_TYPE] .= "NaN"
(size(database_pos[database_pos[:,:MS_TYPE] .== "MS4",:], 1) / size(database_pos,1)) * 100

MS_TYPE_neg = unique(database_neg[:,:MS_TYPE])
database_neg[ismissing.(database_neg[:,:MS_TYPE]),:MS_TYPE] .= "NaN"
(size(database_neg[database_neg[:,:MS_TYPE] .== "MS4",:], 1) / size(database_neg,1)) * 100

FRAGMENTATION_MODE_pos = unique(database_pos[:,:FRAGMENTATION_MODE])
FRAGMENTATION_MODE_neg = unique(database_neg[:,:FRAGMENTATION_MODE])

DATA_TYPE_pos = unique(database_pos[:,:DATA_TYPE])
DATA_TYPE_neg = unique(database_neg[:,:DATA_TYPE])



