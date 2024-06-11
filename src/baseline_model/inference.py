import pandas as pd
import pickle
import boto3

from utils.load_data import load_data
from baseline_model.preprocessing import preprocess_data


def baseline_inference(df: pd.DataFrame) -> pd.DataFrame:
    
    s3 = boto3.resource('s3')
    xgb_model = pickle.loads(s3.Bucket("trainer-injury").Object("baseline_xgb_model.pkl").get()['Body'].read())
    df = preprocess_data(df)
    preds = xgb_model.predict_proba(df[xgb_model.feature_names_in_])[:, 1]
        
    return preds


def main():
    df = load_data()
    df['predictions'] = baseline_inference(df)
    return df


if __name__ == '__main__':
    main()