import pandas as pd
import pickle
import boto3

from baseline_model.load_data import load_data
from baseline_model.preprocessing import preprocess_data


def baseline_inference(df: pd.DataFrame) -> pd.DataFrame:
    
    s3 = boto3.resource('s3')
    xgb_model = pickle.loads(s3.Bucket("trainer-injury").Object("baseline_xgb_model.pkl").get()['Body'].read())
    df = preprocess_data(df)
    df['predictions'] = xgb_model.predict_proba(df[xgb_model.feature_names_in_])[:, 1]
        
    return df


def main():
    df = load_data()
    output = baseline_inference(df)
    return output


if __name__ == '__main__':
    main()