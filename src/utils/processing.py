from typing import Dict
import pandas as pd
import numpy as np
import datetime

from utils.constants import SEED, BADLY_BEATEN_THRESHOLD, LONG_LAYOFF_THRESHOLD
from smoothing.constants import COLS_FOR_SMOOTHING, DENOMINATORS


def get_dnf(df: pd.DataFrame) -> pd.Series:
    dnf = np.where(
            df['length_behind_at_finish'] > 9000,
            1,
            0
        )
    
    return pd.Series(dnf, name='dnf')



def get_scratches(df: pd.DataFrame) -> pd.Series:
    scratched = np.where(
            df['scratch_indicator'] == 'Y',
            1,
            0
        )
    
    vets_scratch = np.where(
        (df['scratch_indicator'] == 'Y') & (df['scratch_reason'].isin(['I', 'J', 'N', 'U', 'V', 'Z'])),
        1,
        0
    )

    return pd.Series(scratched, name='scratched'), pd.Series(vets_scratch, name='vet_scratched')
    

def get_medication(df: pd.DataFrame) -> pd.Series:

    lasix = np.where(
        df['medication'].str.contains('L'),
        1,
        0
    )
    
    bute = np.where(
        df['medication'].str.contains('B'),
        1,
        0
    )

    return pd.Series(lasix, name='lasix'), pd.Series(bute, name='bute')
    

def get_badly_beaten(df: pd.DataFrame) -> pd.Series:
    badly_beaten = np.where(
        (df['length_behind_at_finish'] > BADLY_BEATEN_THRESHOLD) & (df['dnf'] == 0),
        1,
        0
    )

    return pd.Series(badly_beaten, name='badly_beaten')


def get_breakdown(df: pd.DataFrame) -> pd.Series:
    breakdown = np.where(
        (df['long_comment'].str.contains('vanned')) & (df['dnf'] == 1), 
        1,
        0
    )

    return pd.Series(breakdown, name='breakdown')


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df['dnf'] = get_dnf(df)
    df['scratched'], df['vet_scratched'] = get_scratches(df)
    df['lasix'], df['bute'] = get_medication(df)
    df['badly_beaten'] = get_badly_beaten(df)
    df['breakdown'] = get_breakdown(df)

    return df


def group_trainer_info(df: pd.DataFrame) -> pd.DataFrame:

    trainer_entries = df.groupby(['trainer_id', 'trainer_name']).agg({
        'registration_number': 'nunique',
        'race_date': 'count'
    }).reset_index().rename(columns={
        'registration_number': 'n_horses', 
        'race_date': 'n_entries'
    })

    starts = df[df['scratched'] == 0]
    trainer_starts = starts.groupby(['trainer_id', 'trainer_name']).agg({
        'race_date': 'count'
    }).reset_index().rename(columns={
        'race_date': 'n_starts'
    })

    trainer_stats = df.groupby(['trainer_id', 'trainer_name']).agg({
        'dnf': 'sum',
        'scratched': 'sum',
        'vet_scratched': 'sum',
        'badly_beaten': 'sum',
        'breakdown': 'sum'
    }).reset_index()

    trainer_data = trainer_entries.merge(trainer_starts, on=['trainer_id', 'trainer_name'], how='inner')
    trainer_data = trainer_data.merge(trainer_stats, on=['trainer_id', 'trainer_name'], how='inner')
    trainer_data = trainer_data.fillna(0)

    for col in COLS_FOR_SMOOTHING:
        trainer_data[f'{col}_pct'] = trainer_data[col] / trainer_data[DENOMINATORS[col]]


    return trainer_data



def get_prev_race_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df['race_date'] = pd.to_datetime(df['race_date'])
    df = df.sort_values(by=['registration_number', 'race_date'])
    df = df.rename(columns={'distance_id': 'race_distance'})

    df['previous_race_date'] = df.groupby('registration_number')['race_date'].shift(1)
    df['previous_race_dnf'] = df.groupby('registration_number')['dnf'].shift(1)
    df['previous_race_vet_scratched'] = df.groupby('registration_number')['vet_scratched'].shift(1)
    df['previous_race_badly_beaten'] = df.groupby('registration_number')['badly_beaten'].shift(1)
    df['previous_race_breakdown'] = df.groupby('registration_number')['breakdown'].shift(1)
    df['previous_race_scratched'] = df.groupby('registration_number')['scratched'].shift(1)
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
        df['previous_race_scratched'] == 1,
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



def create_train_test_split(
    df: pd.DataFrame, test_size: float, valid_size: float, split_column: str, target_column: str
) -> Dict:
    reg_numbers = list(df[split_column].unique())
    np.random.seed(SEED)
    np.random.shuffle(reg_numbers)

    valid_ids = reg_numbers[: int(valid_size * len(reg_numbers))]
    test_ids = reg_numbers[
        int(valid_size * len(reg_numbers)) : int(
            (valid_size + test_size) * len(reg_numbers)
        )
    ]
    train_ids = reg_numbers[int((valid_size + test_size) * len(reg_numbers)) :]
    cols_to_drop = [target_column, split_column] + [c for c in df.columns if 'target' in c]

    data = {
        'X_train': df[df[split_column].isin(train_ids)].drop(
            columns=cols_to_drop
        ),
        'X_valid': df[df[split_column].isin(valid_ids)].drop(
            columns=cols_to_drop
        ),
        'X_test': df[df[split_column].isin(test_ids)].drop(
            columns=cols_to_drop
        ),
        'y_train': df[df[split_column].isin(train_ids)][target_column],
        'y_valid': df[df[split_column].isin(valid_ids)][target_column],
        'y_test': df[df[split_column].isin(test_ids)][target_column],
    }

    return data


def random_dates(n):
    start_date = datetime.datetime(2019, 1, 1)
    end_date = datetime.datetime(2023, 10, 1)
    delta = end_date - start_date
    random_dates = [start_date + datetime.timedelta(days=np.random.randint(delta.days)) for _ in range(n)]
    
    random_dates = [date.strftime('%Y-%m-%d') for date in random_dates]

    return random_dates
