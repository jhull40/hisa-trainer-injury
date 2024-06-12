from typing import Dict
import pandas as pd
import numpy as np
import datetime

from utils.constants import SEED



def get_dnf(df: pd.DataFrame) -> pd.Series:
    dnf = np.where(
            df['length_behind_at_finish'] > 9000,
            1,
            0
        )
    
    return pd.Series(dnf, name='dnf')



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
