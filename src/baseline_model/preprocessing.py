from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from baseline_model.load_data import load_data
from baseline_model.constants import COLS_FOR_MODEL, TARGET, SEED, TOP_CONDITIONS, TOP_RACE_TYPES


def create_dummy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['race_surface'] = df['surface'].str.replace('InnerTurf', 'Turf').str.replace('InnerDirt', 'Dirt')

    df['race_condition'] = np.where(
        df['track_condition'].isin(TOP_CONDITIONS),
        df['track_condition'],
        None
    )

    df = df.rename(columns={'race_type': 'race_type_raw'})
    df['race_type'] = np.where(
        df['race_type_raw'].isin(TOP_RACE_TYPES),
        df['race_type_raw'],
        None
    )

    df = pd.get_dummies(df, columns=['race_surface', 'race_condition', 'race_type'])
    
    return df


def scale_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = create_dummy_columns(df)
    df = scale_columns(df, ['age'])
    df = df[COLS_FOR_MODEL + [TARGET]]

    return df


def create_train_test_split(df: pd.DataFrame, test_size: float, valid_size: float) -> Dict:
    
    reg_numbers = list(df['registration_number'].unique())
    np.random.seed(SEED)
    np.random.shuffle(reg_numbers)

    valid_ids = reg_numbers[:int(valid_size * len(reg_numbers))]
    test_ids = reg_numbers[int(valid_size * len(reg_numbers)):int((valid_size + test_size) * len(reg_numbers))]
    train_ids = reg_numbers[int((valid_size + test_size) * len(reg_numbers)):]

    data = {
        'X_train': df[df['registration_number'].isin(train_ids)].drop(columns=[TARGET, 'registration_number']),
        'X_valid': df[df['registration_number'].isin(valid_ids)].drop(columns=[TARGET, 'registration_number']),
        'X_test': df[df['registration_number'].isin(test_ids)].drop(columns=[TARGET, 'registration_number']),
        'y_train': df[df['registration_number'].isin(train_ids)][TARGET],
        'y_valid': df[df['registration_number'].isin(valid_ids)][TARGET],
        'y_test': df[df['registration_number'].isin(test_ids)][TARGET]
    }

    return data


if __name__ == '__main__':
    data = load_data()
    data = preprocess_data(data)
    data = create_train_test_split(data, test_size=0.2, valid_size=0.15)
    print(data['X_train'].head())