using CSV, DataFrames, Plots, PyCall, Conda, ProgressBars, FreqTables
pcp = pyimport("pubchempy")
pd = pyimport("padelpy")

raw_data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\all_datasets_unified_IEs_20221121.csv", DataFrame)
#Internal = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\data\\Database_INTERNAL_2022-11-17.csv", DataFrame)

short_data = raw_data[:,["name","SMILES","inchi","pH.aq.","unified_IEs","instrument","source"]]
what_is_this = raw_data[:,["in_model_190714","SPLIT","logIE_pred","inchi","source_brand","unified_IEs"]]
IE_data = raw_data[:,["name","pH_aq","logIE","logIE_pred","unified_IEs"]]

# Insert INCHIKEY and SMILES cols
from_name_data = deepcopy(short_data)
from_name_data.SMILES_cor .= ""
from_name_data.INCHIKEY .= ""
from_inchi_data = deepcopy(from_name_data)

# From inchis
for i in ProgressBar(1:size(from_inchi_data,1))
    try
        cids = pcp.get_compounds(from_inchi_data[i,"inchi"], "inchi")
        shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
        from_inchi_data[i,"SMILES_cor"] = shortest_cid.isomeric_smiles
        from_inchi_data[i,"INCHIKEY"] = shortest_cid.inchikey
    catch
        continue
    end
end
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\from_inchi_data.csv", from_inchi_data)
# From names
for i in ProgressBar(1:size(from_name_data,1))
    try
        cids = pcp.get_compounds(from_name_data[i,"name"], "name")
        shortest_cid = cids[argmin([cids[y].cid for y in 1:length(cids)])]
        from_name_data[i,"SMILES_cor"] = shortest_cid.isomeric_smiles
        from_name_data[i,"INCHIKEY"] = shortest_cid.inchikey
    catch
        continue
    end
end
CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\from_name_data.csv", from_name_data)

from_name_data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\from_name_data.csv", DataFrame)
from_inchi_data = CSV.read("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\from_inchi_data.csv", DataFrame)

missing_from_name = findall(x -> ismissing(x),from_name_data[:,:SMILES_cor])
missing_from_inchi = findall(x -> ismissing(x),from_inchi_data[:,:SMILES_cor])

# Inchi_data covers most compounds except for four. These will be filled in manually
from_inchi_data[missing_from_inchi[1],"SMILES_cor"] = "CC(=NC#N)N(C)CC1=CN=C(C=C1)Cl"
from_inchi_data[missing_from_inchi[1],"INCHIKEY"] = "WCXDHFDTOYPNIE-UHFFFAOYSA-N"
from_inchi_data[missing_from_inchi[2],"SMILES_cor"] = "COC=C(C1=CC=CC=C1OC2=NC=NC(=C2)OC3=CC=CC=C3C#N)C(=O)OC"
from_inchi_data[missing_from_inchi[2],"INCHIKEY"] = "WFDXOXNFNRHQEC-GHRIWEEISA-N"
from_inchi_data[missing_from_inchi[3],"SMILES_cor"] = "C1=CC2=C(C=C1Cl)[N+](=NC(=N2)N3C=CN=C3)[O-]"
from_inchi_data[missing_from_inchi[3],"INCHIKEY"] = "IQGKIPDJXCAMSM-UHFFFAOYSA-N"
deleteat!(from_inchi_data,missing_from_inchi[4])

missing_SMILES = findall(x -> ismissing(x),from_inchi_data[:,:SMILES_cor])          # 3 compounds were corrected, 1 was removed
missing_INCHIKEYs = findall(x -> ismissing(x),from_inchi_data[:,:INCHIKEY])            # There are still missing INCHIKEYs

deleteat!(from_inchi_data,missing_INCHIKEYs)         # Let's just delete them (You can make separate datasets for the SMILES and CNL models)
missing_SMILES = findall(x -> ismissing(x),from_inchi_data[:,:SMILES_cor])          # 3 compounds were corrected, 1 was removed
missing_INCHIKEYs = findall(x -> ismissing(x),from_inchi_data[:,:INCHIKEY])            # There are still missing INCHIKEYs

CSV.write("C:\\Users\\alex_\\Documents\\GitHub\\IE_prediction-project\\Unified\\data\\filtered_data.csv", from_inchi_data)