from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta

from smoothing.constants import (
    MIN_STARTS_THRESHOLD,
    INITIAL_PARAMS,
    BOUNDS,
    N_SAMPLES,
    N_TRIALS,
    COLS_FOR_SMOOTHING,
    DENOMINATORS,
)


def group_trainer_info(df: pd.DataFrame) -> pd.DataFrame:

    trainer_entries = df.groupby('trainer_id').agg({
        'registration_number': 'nunique',
        'race_date': 'count'
    }).reset_index().rename(columns={
        'registration_number': 'n_horses', 
        'race_date': 'n_entries'
    })

    starts = df[df['scratched'] == 0]
    trainer_starts = starts.groupby('trainer_id').agg({
        'race_date': 'count'
    }).reset_index().rename(columns={
        'race_date': 'n_starts'
    })

    trainer_stats = df.groupby('trainer_id').agg({
        'dnf': 'sum',
        'scratched': 'sum',
        'vet_scratched': 'sum',
        'badly_beaten': 'sum',
        'breakdown': 'sum'
    }).reset_index()

    trainer_data = trainer_entries.merge(trainer_starts, on='trainer_id', how='inner')
    trainer_data = trainer_data.merge(trainer_stats, on='trainer_id', how='inner')
    trainer_data = trainer_data.fillna(0)

    for col in COLS_FOR_SMOOTHING:
        trainer_data[f'{col}_pct'] = trainer_data[col] / trainer_data[DENOMINATORS[col]]


    return trainer_data


def log_likelihood(params: Tuple[float], x: pd.Series, total: pd.Series) -> float:
    alpha0, beta0 = params
    log_prob = (
        betaln(x + alpha0, total - x + beta0)
        - betaln(alpha0, beta0)
        - betaln(x + 1, total - x + 1)
    )

    return -np.sum(log_prob)


def calculate_beta_binom_params(df: pd.DataFrame, x_column: str, total_column: str) -> Tuple[float, float]:
    
    df_threshold = df[df["n_starts"] >= MIN_STARTS_THRESHOLD]

    x = df_threshold[x_column].values
    total = df_threshold[total_column].values

    result = minimize(
        log_likelihood,
        INITIAL_PARAMS,
        args=(x, total),
        method="L-BFGS-B",
        bounds=BOUNDS,
    )

    alpha0, beta0 = result.x

    return alpha0, beta0


def get_smoothing_params(df: pd.DataFrame) -> Dict:
    smoothing_params = {}
    for col in COLS_FOR_SMOOTHING:
        alpha0, beta0 = calculate_beta_binom_params(df, f'{col}', DENOMINATORS[col])
        smoothing_params[col] = {
            'alpha0': alpha0,
            'beta0': beta0
        }

    with open('output/smoothing/smoothing_params.json', 'w') as f:
        json.dump(smoothing_params, f)

    return smoothing_params
    

def calculate_smoothed_rates(df: pd.DataFrame, smoothing_params: Dict) -> pd.DataFrame:
    for c in COLS_FOR_SMOOTHING:
        alpha0 = smoothing_params[c]['alpha0']
        beta0 = smoothing_params[c]['beta0']

        df[f'{c}_smoothed'] = (df[c] + alpha0) / (df[DENOMINATORS[c]] + alpha0 + beta0)

    return df


def generate_beta_binomial_samples(
    n: int, alpha0: float, beta0: float, size: int
) -> np.ndarray:
    p = np.random.beta(alpha0, beta0, size)
    samples = np.random.binomial(n, p, size)

    return samples


