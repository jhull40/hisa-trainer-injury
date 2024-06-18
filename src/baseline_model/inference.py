import pandas as pd
import pickle
import boto3

from utils.load_data import load_data
from utils.processing import build_features, get_prev_race_features
from baseline_model.preprocessing import preprocess_data


def baseline_inference(races: pd.DataFrame) -> pd.DataFrame:
    
    s3 = boto3.resource('s3')
    xgb_model = pickle.loads(s3.Bucket("trainer-injury").Object("baseline_xgb_model.pkl").get()['Body'].read())
    
    df = build_features(races)
    df, cols_for_model = preprocess_data(df)
    df = get_prev_race_features(df)
    preds = xgb_model.predict(df[xgb_model.feature_names_in_])                
    
    return preds



def group_baseline_predictions(races: pd.DataFrame, trainers: pd.DataFrame) -> pd.DataFrame:
    races['baseline_prediction'] = baseline_inference(races)
    trainer_baseline = races.groupby(['trainer_id', 'trainer_name']).agg({
        'baseline_prediction': 'sum'
    }).reset_index()
    
    trainers = trainers.merge(trainer_baseline, on=['trainer_id', 'trainer_name'], how='inner')
        
    return trainers


def main():
    df = load_data(False, True)
    df['predictions'] = baseline_inference(df)
    return df


if __name__ == '__main__':
    main()