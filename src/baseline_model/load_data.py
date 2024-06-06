import pandas as pd
from baseline_model.constants import YEARS_TO_MODEL, DATA_BUCKET


def load_data(local=False):
    if local:
        df = pd.read_csv('/users/jameshull/documents/github/hisa-data/races_2023.csv', nrows=1000)
    
    else:
        df = pd.DataFrame()
        for yr in YEARS_TO_MODEL:
            data_key = f'races_{yr}.csv' 
            data_location = 's3://{}/{}'.format(DATA_BUCKET, data_key) 
            df_yr = pd.read_csv(data_location) 
            df = pd.concat([df, df_yr], ignore_index=True)

    
    return df


if __name__ == "__main__":
    data = load_data()
    print(data.head())
