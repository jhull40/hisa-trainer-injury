from typing import Tuple, List
import pandas as pd
import numpy as np
from scipy.stats import lognorm
import datetime
import warnings
warnings.simplefilter("ignore")

from baseline_model.inference import baseline_inference
from utils.processing import get_dnf
from trainer_risk.constants import (
    BADLY_BEATEN_THRESHOLD, 
    LONG_LAYOFF_THRESHOLD, 
    MIN_ENTRIES_TARGET,
    FEATURE_DAY_DELTAS,
    TARGET_DAY_DELTA,
    N_ENTRIES_SAMPLES
)


def get_smoothing_params() -> Tuple[float, float]:
    ALPHA0 = 1
    BETA0 = 1

    return ALPHA0, BETA0


def get_xDNF(df: pd.DataFrame) -> pd.DataFrame:
    scratches = df[df['scratch_indicator'] == 'Y']
    scratches['xDNF'] = np.nan
    racers = df[df['scratch_indicator'] != 'Y']
    racers['xDNF'] = baseline_inference(df)

    df = pd.concat([racers, scratches])
    
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df['dnf'] = get_dnf(df)

    df['scratched'] = np.where(
        df['scratch_indicator'] == 'Y',
        1,
        0
    )
    
    df['lasix'] = np.where(
        df['medication'].str.contains('L'),
        1,
        0
    )
    
    df['bute'] = np.where(
        df['medication'].str.contains('B'),
        1,
        0
    )
    
    df['badly_beaten'] = np.where(
        (df['length_behind_at_finish'] > BADLY_BEATEN_THRESHOLD) & (df['length_behind_at_finish'] < 9000),
        1,
        0
    )
    
    return df
    
def get_prev_race_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df.sort_values(by=['registration_number', 'race_date'])
    df = df.rename(columns={'distance_id': 'race_distance'})

    df['previous_race_date'] = df.groupby('registration_number')['race_date'].shift(1)
    df['previous_race_dnf'] = df.groupby('registration_number')['dnf'].shift(1)
    df['previous_race_scratch'] = df.groupby('registration_number')['scratched'].shift(1)
    df['previous_race_distance'] = df.groupby('registration_number')['race_distance'].shift(1)
    df['previous_surface'] = df.groupby('registration_number')['surface'].shift(1)
    df['days_since_last_race'] = (df['race_date'] - df['previous_race_date']).dt.days


    df['distance_delta'] = df['race_distance'] - df['previous_race_distance']
    df['distance_jump'] = np.where(
        df['distance_delta'] > 200,
        1,
        0
    )

    df['rest_after_dnf'] = np.where(
        df['previous_race_dnf'] == 1,
        df['days_since_last_race'],
        np.nan
    )

    df['rest_after_scratch'] = np.where(
        df['previous_race_scratch'] == 1,
        df['days_since_last_race'],
        np.nan
    )

    df['surface_change'] = np.where(
        df['surface'] != df['previous_surface'],
        1,
        0
    )

    df['long_layoff'] = np.where(
        df['days_since_last_race'] > LONG_LAYOFF_THRESHOLD,
        1,
        0
    )
    
    return df


def fit_params(grouped_df: pd.DataFrame) -> pd.Series:
    try:
        params = lognorm.fit(grouped_df['days_since_last_race'].dropna())
    except Exception as e:
        # print(e)
        params = (None, None, None)

    trainer_params = pd.Series(params, index=['lognorm_p1', 'lognorm_p2', 'lognorm_p3'])
    
    return trainer_params


def get_first_long(df: pd.DataFrame) -> pd.DataFrame:
    
    first_long = df[df['race_distance'] > 800].sort_values(by=['race_date']).groupby(['registration_number', 'trainer_id']).first().reset_index()

    trainer_first_long = first_long.groupby(['trainer_id']).agg({
            'age': 'median'
        }).reset_index().rename(columns={'age': 'first_long_age'})

    return trainer_first_long


def group_trainer_data(df: pd.DataFrame) -> pd.DataFrame:
    alpha0, beta0 = get_smoothing_params()

    trainers = df.groupby(['trainer_id']).agg({
        'race_number': 'count',
        'registration_number': 'nunique',
        'scratched': 'sum',
        'dnf': 'sum',
        'xDNF': 'sum',
        'age': 'min',
        'lasix': 'sum',
        'bute': 'sum',
        'days_since_last_race': ['min', 'median'],
        'rest_after_dnf': 'median',
        'rest_after_scratch': 'median',
        'distance_jump': 'sum',
        'surface_change': 'sum',
        'long_layoff': 'sum',
        'badly_beaten': 'sum',
    }).reset_index()

    trainers.columns = ['trainer_id',
    'n_entries', 'unique_horses', 'scratched', 'dnf', 'xDNF', 'min_age', 'lasix', 'bute', 'days_since_last_race_min', 'days_since_last_race_median', 
    'rest_after_dnf_median', 'rest_after_scratch_median', 'distance_jump', 'surface_changes', 'long_layoffs',
    'badly_beaten'
    ]

    trainers['scratches_per_entry'] = trainers['scratched'] / trainers['n_entries']
    trainers['dnf_per_entry'] = trainers['dnf'] / trainers['n_entries']
    trainers['dnf_smoothed'] = (trainers['dnf'] + alpha0) / (trainers['n_entries'] + alpha0 + beta0)
    trainers['dnfAA'] = trainers['dnf'] - trainers['xDNF']
    
    trainers['badly_beaten_pct'] = trainers['badly_beaten'] / trainers['n_entries']
    trainers['lasix_pct'] = trainers['lasix'] / trainers['n_entries']
    trainers['bute_pct'] = trainers['bute'] / trainers['n_entries']
    
    
    return trainers


def create_features(df: pd.DataFrame, suffix: str = None) -> pd.DataFrame:

    df = extract_features(df)
    df = get_prev_race_features(df)
    trainer_params = df.groupby('trainer_id').apply(fit_params)
    first_long = get_first_long(df)
    trainers = group_trainer_data(df)
    trainers = trainers.merge(first_long, on=['trainer_id'], how='left')
    trainers = trainers.merge(trainer_params, on=['trainer_id'], how='left')
    
    for c in trainers.columns:
        if 'lognorm' in c:
            trainers[c] = round(trainers[c], 5)
    
    if suffix:
        trainers.columns = [c + f'_{suffix}' for c in trainers.columns]
        trainers = trainers.rename(columns={
            f'trainer_id_{suffix}': 'trainer_id'
        })
    
    return trainers


def create_targets(df: pd.DataFrame): 

    alpha0, beta0 = get_smoothing_params()
    df = extract_features(df)
    
    trainers = df.groupby('trainer_id').agg({
        'race_number': 'count',
        'dnf': 'sum',
        'scratched': 'sum',
        'badly_beaten': 'sum',
        #'long_layoff': 'sum',
    })
    
    trainers = trainers.rename(columns={
        'race_number': 'n_entries'
    }).reset_index()
    
    trainers['scratches_per_entry'] = trainers['scratched'] / trainers['n_entries']
    trainers['dnf_per_entry'] = trainers['dnf'] / trainers['n_entries']
    trainers['badly_beaten_pct'] = trainers['badly_beaten'] / trainers['n_entries']
    trainers['dnf_smoothed'] = (trainers['dnf'] + alpha0) / (trainers['n_entries'] + alpha0 + beta0)
    
    trainers = trainers[['trainer_id', 'n_entries', 'dnf_per_entry', 'dnf_smoothed']].rename(columns={
        'dnf_per_entry': 'target',
        'n_entries': 'target_n_entries',
        'dnf_smoothed': 'target_dnf_smoothed',
    })
    
    trainers = trainers[trainers['target_n_entries'] >= MIN_ENTRIES_TARGET]
    
    return trainers


def create_full_dataset_by_date(df: pd.DataFrame, prediction_dates: List[str]) -> pd.DataFrame:

    full_model_df = pd.DataFrame()
    for prediction_date in prediction_dates:
        df_date = pd.DataFrame()
        for feature_day_delta in FEATURE_DAY_DELTAS:
            feature_end_date = datetime.datetime.strptime(prediction_date, '%Y-%m-%d')
            feature_start_date = feature_end_date - datetime.timedelta(days=feature_day_delta)

            target_start_date = feature_end_date + datetime.timedelta(days=1)
            target_end_date = target_start_date + datetime.timedelta(days=TARGET_DAY_DELTA)

            df_features = df[(df['race_date'] >= feature_start_date) & (df['race_date'] <= feature_end_date)]
            df_target = df[(df['race_date'] >= target_start_date) & (df['race_date'] <= target_end_date)]

            features = create_features(df_features, str(feature_day_delta))
            targets = create_targets(df_target)

            if df_date.empty:
                df_date = features.merge(targets, how='inner', on='trainer_id')
            else:
                df_date_tmp = features.merge(targets, how='inner', on='trainer_id')
                df_date = df_date.merge(df_date_tmp, how='inner', on=['trainer_id', 'target_n_entries', 'target', 'target_dnf_smoothed'])
                
        
        full_model_df = pd.concat([full_model_df, df_date], ignore_index=True)

    full_model_df.to_csv(f's3://trainer-injury/full_dataset_by_date.csv', index=False)

    return full_model_df
        

def create_full_dataset_by_entries(df: pd.DataFrame, prediction_dates: List[str]) -> pd.DataFrame:

    for prediction_date in prediction_dates:
        df_date = pd.DataFrame()
        for n_entries in N_ENTRIES_SAMPLES:
            feature_end_date = datetime.datetime.strptime(prediction_date, '%Y-%m-%d')
            feature_start_date = feature_end_date - datetime.timedelta(days=365)
            target_end_date = feature_end_date + datetime.timedelta(days=365)
            
            df_past = df[(df['race_date'] < prediction_date) & (df['race_date'] >= feature_start_date)]
            df_future = df[(df['race_date'] >= prediction_date) & (df['race_date'] < target_end_date)]
            
            df_features = df_past.groupby('trainer_id').apply(lambda x: x.tail(n_entries)).reset_index(drop=True)
            df_target = df_future.groupby('trainer_id').apply(lambda x: x.tail(250)).reset_index(drop=True)

            features = create_features(df_features, str(n_entries))
            targets = create_targets(df_target)

            if df_date.empty:
                df_date = features.merge(targets, how='inner', on='trainer_id')
            else:
                df_date_tmp = features.merge(targets, how='inner', on='trainer_id')
                df_date = df_date.merge(df_date_tmp, how='inner', on=['trainer_id', 'target_n_entries', 'target', 'target_dnf_smoothed'])
                
        
        full_model_df = pd.concat([full_model_df, df_date], ignore_index=True)
        
    full_model_df.to_csv(f's3://trainer-injury/full_dataset_by_entries.csv', index=False)

    return full_model_df


def create_full_dataset(df: pd.DataFrame, prediction_dates: List[str], method: str) -> pd.DataFrame:
    if method == 'date':
        return create_full_dataset_by_date(df, prediction_dates)
    elif method == 'entries':
        return create_full_dataset_by_entries(df, prediction_dates)
    else:
        raise ValueError('Method must be either "date" or "entries"')