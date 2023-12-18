using CSV, DataFrames, ProgressBars, Plots

harry = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\data_for_Alex_from_Harry.csv",DataFrame)
helen = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\data_for_Alex_from_Helen.csv",DataFrame)

# First, calculate the SMILES IE
# Second, calculate the CNL IE
