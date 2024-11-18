import os
import zipfile
import pandas as pd
import shutil
import pickle

"""
    It opens all the zip files in the directory and extracts the CSV files. It creates a unique dataframe with all the data
"""

# Define the directory where the zip files are located
zip_dir = 'Pendolarismo per sezione di censimento/'

# List to hold all the dataframes
dataframes_list = []

print("Extracting data from zip files...")
# Iterate over each file in the directory
for filename in os.listdir(zip_dir):
    # Check if the file is a zip file
    if filename.endswith('.zip'):
        # Construct the full path to the file
        file_path = os.path.join(zip_dir, filename)
        # Extract the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Create the folder
            folder_name = filename.replace('.zip', '')
            os.mkdir(os.path.join(zip_dir, folder_name))
            # Extract the contents into the folder
            zip_ref.extractall(os.path.join(zip_dir, folder_name))
            # Find the CSV file (which has the same name as the zip file)
            csv_filename = filename.replace('.zip', '.csv')
            folder_name = filename.replace('.zip', '')
            folder_path = os.path.join(zip_dir, folder_name)
            csv_path = os.path.join(folder_path, csv_filename)
            # Parse the CSV file into a pandas dataframe,
            # We remove columns 'NOMEASC_ORIG' and 'NOMEASC_DEST' because they are not necessary
            # and they contain special characters that are not supported by the encoding
            df = pd.read_csv(csv_path,
                             low_memory=False,
                             encoding='ISO-8859-1',
                             sep=';').drop(['NOMEASC_ORIG', 'NOMEASC_DEST'], axis=1)
            dataframes_list.append(df)
            # Delete the folder and its contents
            shutil.rmtree(folder_path)

# Concatenate all the dataframes into a single dataframe
data = pd.concat(dataframes_list)
print("Data extracted successfully")
print("Preprocessing data...")

# process data and save
columns_to_remove = ['REGIONE_ORIG', 'REGIONE_DEST', 'PROVINCIA_ORIG', 'PROVINCIA_DEST', 'COMUNE_ORIG', 'COMUNE_DEST',
                     'CODLOC_ORIG', 'CODLOC_DEST', 'LOCALITA_ORIG', 'LOCALITA_DEST', 'CODASC_ORIG', 'CODASC_DEST',
                     'CODCOM_ORIG', 'CODCOM_DEST']
data.drop(columns_to_remove, axis=1, inplace=True)
columns_to_remove = ['TOTALE_STUDENTI', 'TOTALE_LAVORATORI']
data.drop(columns_to_remove, axis=1, inplace=True)
# drop also levels related to ACE and NSEZ (so get only Region, Province, Municipality)
columns_to_remove = ['ACE_ORIG', 'ACE_DEST', 'NSEZ_ORIG', 'NSEZ_DEST']
data.drop(columns_to_remove, axis=1, inplace=True)
# change names of columns
columns = ['LEVEL1_ORIG', 'LEVEL2_ORIG', 'LEVEL3_ORIG', 'LEVEL1_DEST', 'LEVEL2_DEST', 'LEVEL3_DEST', 'COUNT']
data.columns = columns
# perform a groupby
data = data.groupby(columns, as_index=False)['COUNT'].sum().reset_index(drop=True)
# create unique identifier adding the level number
data['LEVEL1_ORIG'] = data['LEVEL1_ORIG'].astype(str) + '_1'
data['LEVEL2_ORIG'] = data['LEVEL2_ORIG'].astype(str) + '_2'
data['LEVEL3_ORIG'] = data['LEVEL3_ORIG'].astype(str) + '_3'
data['LEVEL1_DEST'] = data['LEVEL1_DEST'].astype(str) + '_1'
data['LEVEL2_DEST'] = data['LEVEL2_DEST'].astype(str) + '_2'
data['LEVEL3_DEST'] = data['LEVEL3_DEST'].astype(str) + '_3'
data['COUNT'] = data['COUNT'].astype(int)
# add root
data['LEVEL0_ORIG'] = [0] * len(data)
data['LEVEL0_DEST'] = [0] * len(data)
# reorder columns
data = data[['LEVEL0_ORIG', 'LEVEL1_ORIG', 'LEVEL2_ORIG', 'LEVEL3_ORIG', 'LEVEL0_DEST', 'LEVEL1_DEST', 'LEVEL2_DEST',
             'LEVEL3_DEST', 'COUNT']]

# saving the preprocessed data
folder_path = "../data/Italy"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
data.to_csv(f"{folder_path}/data.csv", index=False)

print("Data preprocessed and saved successfully")

"""
    Constructing the geographical spine from data, it is nested dictionary
"""

print("Constructing the geographical spine...")


def create_dictionary(set):
    dictionary = {}
    for element in set:
        dictionary[element] = {}
    return dictionary


geo_spine = {}
# add region level
regions = set(data['LEVEL1_ORIG']).union(set(data['LEVEL1_DEST']))
for region in regions:
    geo_spine[region] = {}
    # add province level
    df_origin_region = data[(data['LEVEL1_ORIG'] == region)]
    df_destination_region = data[(data['LEVEL1_DEST'] == region)]
    provinces = set(df_origin_region['LEVEL2_ORIG']).union(set(df_destination_region['LEVEL2_DEST']))
    # update the dictionary
    geo_spine[region] = create_dictionary(provinces)
    # add municipality level
    for province in provinces:
        df_origin_province = df_origin_region[(df_origin_region['LEVEL2_ORIG'] == province)]
        df_destination_province = df_destination_region[(df_destination_region['LEVEL2_DEST'] == province)]
        municipalities = set(df_origin_province['LEVEL3_ORIG']).union(set(df_destination_province['LEVEL3_DEST']))
        # update the dictionary
        geo_spine[region][province] = create_dictionary(municipalities)


# save the geo_spine using pickle
folder_path = "../data/Italy/structure"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(f'{folder_path}/geo_spine.pickle', 'wb') as handle:
    pickle.dump(geo_spine, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Geographical spine constructed and saved successfully")
