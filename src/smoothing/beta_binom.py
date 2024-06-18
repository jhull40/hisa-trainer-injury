from typing import Tuple, Dict
import json
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
    DISPLAY_NAMES,
)


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


def load_smoothing_params() -> Dict:
    with open('output/smoothing/smoothing_params.json', 'r') as f:
        smoothing_params = json.load(f)

    return smoothing_params


def calculate_smoothed_rates(df: pd.DataFrame, smoothing_params: Dict) -> pd.DataFrame:
    for c in COLS_FOR_SMOOTHING:
        alpha0 = smoothing_params[c]['alpha0']
        beta0 = smoothing_params[c]['beta0']

        df[f'{c}_pct_smoothed'] = (df[c] + alpha0) / (df[DENOMINATORS[c]] + alpha0 + beta0)

    return df
    


def generate_beta_binomial_samples(
    n: int, alpha0: float, beta0: float, size: int
) -> np.ndarray:
    p = np.random.beta(alpha0, beta0, size)
    samples = np.random.binomial(n, p, size)

    return samples


def create_sample_plot(alpha0: float, beta0: float, column_name: str) -> None:
    samples = generate_beta_binomial_samples(N_TRIALS, alpha0, beta0, N_SAMPLES)

    plt.figure()
    plt.hist(
        samples,
        bins=np.arange(0, N_TRIALS / 10) - 0.5,
        density=False,
        alpha=0.75,
        color="blue",
        edgecolor="black",
    )

    plt.xlim(-0.05, max(samples) + 0.1*max(samples))
    plt.xlabel(f"{DISPLAY_NAMES[column_name]} per {N_TRIALS} starts")
    plt.ylabel("Count of Trainers")
    plt.title(f"Simulated Number of {DISPLAY_NAMES[column_name]}")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"output/smoothing/{column_name}_simulation_per_1000.png")
    plt.close()




def create_ranking_plot(df: pd.DataFrame, alpha0: float, beta0: float, column_name: str) -> None:
    df = df.dropna(subset=[f"{column_name}_pct", DENOMINATORS[column_name], f"{column_name}_pct_smoothed"])

    df["low"] = beta.ppf(0.025, alpha0 + df[column_name], beta0 + df[DENOMINATORS[column_name]] - df[column_name])
    df["high"] = beta.ppf(0.975, alpha0 + df[column_name], beta0 + df[DENOMINATORS[column_name]] - df[column_name])

    if df.shape[0] > 10:
        print("Warning: Too many trainers to plot. Showing a random sample of 10.")
        df = df.sample(10)

    df = df.sort_values(f'{column_name}_pct_smoothed').reset_index(drop=True)

    plt.figure()
    plt.errorbar(
        df[f'{column_name}_pct_smoothed'],
        df["trainer_name"].astype(str),
        xerr=[df[f'{column_name}_pct_smoothed'] - df["low"], df["high"] - df[f'{column_name}_pct_smoothed']],
        fmt="o",
        color="blue",
        ecolor="black",
        capsize=3,
    )
    plt.axvline(x=alpha0 / (alpha0 + beta0), color="red", linestyle="--")
    plt.xlabel(f"Smoothed {DISPLAY_NAMES[column_name]} Percentage, with 95% credible interval")
    plt.ylabel("Trainer")
    plt.title(f"Trainer Rankings by Smoothed {DISPLAY_NAMES[column_name]} Percentage")
    plt.tight_layout()
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.savefig(f"output/smoothing/{column_name}_ranking_plot.png")
    plt.close()


def create_smoothed_scatter_plot(df: pd.DataFrame, alpha0: float, beta0: float, column_name: str) -> None:

    df = df.dropna(subset=[f"{column_name}_pct", DENOMINATORS[column_name], f"{column_name}_pct_smoothed"])
    plt.figure()
    plt.scatter(
        df[f"{column_name}_pct"],
        df[f"{column_name}_pct_smoothed"],
        c=np.log(df[DENOMINATORS[column_name]]),
        cmap="bwr",
        alpha=0.5,
    )
    if column_name == "breakdown":
        min_val = -0.0005
    else:
        min_val = -0.005

    max_val = max(df[f'{column_name}_pct_smoothed']) + 0.01*max(df[f'{column_name}_pct_smoothed'])
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", alpha=0.8)
    plt.plot(
        [0, max_val],
        [(alpha0) / (alpha0 + beta0), (alpha0) / (alpha0 + beta0)],
        color="black",
        linestyle="--",
    )
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(f"Observed {DISPLAY_NAMES[column_name]} Rate")
    plt.ylabel(f"Smoothed {DISPLAY_NAMES[column_name]} Rate")
    plt.title(f"Observed vs. Smoothed {DISPLAY_NAMES[column_name]} Rate")
    plt.colorbar(label="Log Number of Starts")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"output/smoothing/{column_name}_smoothed_scatter.png")
    plt.close()
