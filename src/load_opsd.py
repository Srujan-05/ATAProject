import pandas as pd
import numpy as np
import re
import os

def preprocess_data(source_taget="./data/time_series_60min_singleindex_filtered.csv", file_name="_preprocessed_data.csv", file_directory="data", regions=["AT", "CH", "FR"]):
    file_path = os.path.join(".", file_directory)
    three_regions_data = pd.read_csv(source_taget)

    split_regions_data = {region:["utc_timestamp"] for region in regions}

    data_columns = three_regions_data.columns

    for column in data_columns:
        for region in regions:
            if re.match(region, column):
                # print("Region: {}, data: {}".format(region, column))
                split_regions_data[region].append(column)

    for region in regions:
        # print(three_regions_data[split_regions_data[region]])
        data_frame = three_regions_data[split_regions_data[region]]
        data_frame = data_frame.rename(columns={"utc_timestamp": "timestamp"})
        data_frame = data_frame.rename(columns={columns:columns[3:] for columns in split_regions_data[region]})
        # print(data_frame)
        for column in data_frame.columns[1:]:
            if data_frame.loc[:, column].isna().any():
                data_frame.loc[:, column] = data_frame.loc[:, column].replace(np.nan, data_frame.loc[:, column].mean())
        print(os.path.join(file_path, region+file_name))
        data_frame.to_csv(os.path.join(file_path, region+file_name), index=False)
        print("Completed adding {}'s data to {}!".format(region, file_path))

def load_data(countryCode, dataDir="data"):
    # dataDir = os.path.join("..", dataDir)
    if os.path.isdir(dataDir):
        files = os.listdir(dataDir)
        for file in files:
            if re.match(countryCode, file):
                return pd.read_csv(os.path.join(dataDir, file))
        preprocess_data(source_taget=os.path.join(dataDir, "time_series_60min_singleindex_filtered.csv"),file_directory=dataDir)
        files = os.listdir(dataDir)
        for file in files:
            if re.match(countryCode, file):
                return pd.read_csv(os.path.join(dataDir, file))
        print("Country Code not available!!!")
        return None
    else:
        print("Data Path doesn't exist to load data!!!")
        return None


if __name__ == "__main__":
    load_data("FR")