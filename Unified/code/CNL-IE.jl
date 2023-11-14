using CSV, DataFrames, Plots, PyCall, Conda, ProgressBars, FreqTables
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

# Load data
data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", DataFrame)
Internal_raw = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)

# Create a shorter (filtered) Internal database with only the relevant compounds
common_INCHIKEYS_indx = findall(x -> x in unique(data[:,"INCHIKEY"]), Internal_raw[:,:INCHIKEY])
Internal_filtered = Internal_raw[common_INCHIKEYS_indx,:]
common_INCHIKEYS_indx = findall(x -> x in unique(data[:,"INCHIKEY"]), Internal_filtered[:,:INCHIKEY])
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", Internal_filtered)
Internal_filtered = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\Database_INTERNAL_filtered.csv", DataFrame)

