import pandas as pd
import numpy as np

from utils.processing import get_dnf
from baseline_model.constants import (
    SURFACES,
    RACE_TYPES,
    COURSE_TYPES,
    TRACK_CONDITIONS,
    WEATHERS,
    TRACK_SEALEDS,
    SEXES,
    TARGET
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

    df['dnf'] = get_dnf(df)
    df = df[df['scratch_indicator'] == 'N']
    df = df[cols_for_model]

    return df

