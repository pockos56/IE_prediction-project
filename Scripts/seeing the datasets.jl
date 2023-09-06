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



ESI_name = "neg"
ESI_name = "pos"

# Load data files
amide = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_amide_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
norman = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_norman_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
MB = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_MB_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
NIST = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction\\data\\CNL-IE datasets\\CNLIE_NIST_$(ESI_name)_selectedCNLs_20mDabin.csv", DataFrame)
data_whole_raw = vcat(vcat(vcat(norman, amide), MB), NIST)

### FINGERPRINTS OPTIMAZATION ###
    # Set ranges for hyperparameters
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(20:40:80), vcat(collect(100:100:900),collect(1000:200:1400)))
    random_seed_r = collect(1:1:3)
    learn_r = vcat(collect(0.02:0.02:0.15), collect(0.2:0.2:0.8))


    # Set ranges for hyperparameters
    leaf_r = collect(4:2:10)
    tree_r = vcat(collect(200:200:800),(collect(1000:200:1600)))
    random_seed_r = collect(1:3)
    learn_r = [0.01,0.05,0.1,0.15,0.2]
    consensus_threshold_r = [0, 0.1, 0.2, 0.25]
    range_ = collect(0:1:5)
    consensus_algorithm_r = ["removal","replacement"]
    removeminusones_r = [true,false]


