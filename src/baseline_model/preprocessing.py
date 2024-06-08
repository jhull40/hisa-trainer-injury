from typing import Dict
import pandas as pd
import numpy as np

from baseline_model.load_data import load_data
from baseline_model.constants import (
    SURFACES,
    RACE_TYPES,
    COURSE_TYPES,
    TRACK_CONDITIONS,
    WEATHERS,
    TRACK_SEALEDS,
    SEXES,
    TARGET,
    SEED
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    cols_for_model = ['age', 'registration_number', TARGET]
    for i in SURFACES:
        df[f'surface_{i}'] = np.where(
            df['surface'] == i, 1, 0
        )
        cols_for_model.append(f'surface_{i}')

    for i in RACE_TYPES:
        df[f'race_type_{i}'] = np.where(
            df['race_type'] == i, 1, 0
        )
        cols_for_model.append(f'race_type_{i}')

    for i in COURSE_TYPES:
        df[f'course_type_{i}'] = np.where(
            df['course_type'] == i, 1, 0
        )
        cols_for_model.append(f'course_type_{i}')

    for i in TRACK_CONDITIONS:
        df[f'track_condition_{i}'] = np.where(
            df['track_condition'] == i, 1, 0
        )
        cols_for_model.append(f'track_condition_{i}')

    for i in WEATHERS:
        df[f'weather_{i}'] = np.where(
            df['weather'] == i, 1, 0
        )
        cols_for_model.append(f'weather_{i}')

    for i in TRACK_SEALEDS:
        df[f'track_sealed_{i}'] = np.where(
            df['track_sealed_indicator'] == i, 1, 0
        )
        cols_for_model.append(f'track_sealed_{i}')

    for i in SEXES:
        df[f'sex_{i}'] = np.where(
            df['sex'] == i, 1, 0
        )
        cols_for_model.append(f'sex_{i}')

    df['dnf'] = np.where(
        (df['trouble_indicator'] == 'Y') | (df['length_behind_at_finish'] == 9999),
        1,
        0
    )

    df = df[df['scratch_indicator'] == 'N']
    df = df[cols_for_model]

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

    data = {
        'X_train': df[df[split_column].isin(train_ids)].drop(
            columns=[target_column, split_column]
        ),
        'X_valid': df[df[split_column].isin(valid_ids)].drop(
            columns=[target_column, split_column]
        ),
        'X_test': df[df[split_column].isin(test_ids)].drop(
            columns=[target_column, split_column]
        ),
        'y_train': df[df[split_column].isin(train_ids)][target_column],
        'y_valid': df[df[split_column].isin(valid_ids)][target_column],
        'y_test': df[df[split_column].isin(test_ids)][target_column],
    }

    return data

