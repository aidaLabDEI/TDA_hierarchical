import pandas as pd
import os
import pickle

# import data
file_path = "../../data/Ireland/Commuting_data.csv"
data = pd.read_csv(file_path, encoding='latin1')
# remove columns that are not necessary
columns = ["RESIDENCE_COUNTY", "RESIDENCE_CSOED", "POWSC_COUNTY","POWSC_CSOED",  "COUNT"]
data = data[columns]
# rename columns
data.columns = ["LEVEL1_ORIG", "LEVEL2_ORIG", "LEVEL1_DEST", "LEVEL2_DEST", "COUNT"]

# create the spine according to what we see in the dataset
spine = dict()
columns_origin = ["LEVEL1_ORIG", "LEVEL2_ORIG"]
columns_destination = ["LEVEL1_DEST", "LEVEL2_DEST"]
county_combinations = set(data[columns_origin[0]]).union(set(data[columns_destination[0]]))
for county in county_combinations:
    spine[county] = dict()
    df_county = data[data[columns_origin[0]] == county]
    csoed_set = set(df_county[columns_origin[1]])
    df_county = data[data[columns_destination[0]] == county]
    csoed_set = csoed_set.union(set(df_county[columns_destination[1]]))
    for csoed in csoed_set:
        spine[county][csoed] = {}

# save the spine
folder_path = "../../data/Ireland/structure"
file_name_1 = f"{folder_path}/geo_spine.pickle"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(file_name_1, "wb") as f:
    pickle.dump(spine, f)

# save the dataset
folder_path = "../../data/Ireland"
file_name_2 = f"{folder_path}/data.csv"
data.to_csv(file_name_2, index=False)

print("Data Preprocessed Successfully")
print("Spine saved successfully in:  ", file_name_1)
print("Dataset saved successfully in:  ", file_name_2)


