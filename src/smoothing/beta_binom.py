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
)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = (
        df.groupby("trainer_name")
        .agg(
            {
                "dnf": "sum",
                "registration_number": "nunique",
                "ChartID": "count",
            }
        )
        .reset_index()
        .rename(columns={"registration_number": "n_horses", "ChartID": "n_starts"})
    )

    df["dnf_pct"] = df["dnf"] / df["n_starts"]

    return df


def log_likelihood(params: Tuple[float], x: pd.Series, total: pd.Series) -> float:
    alpha, beta = params
    log_prob = (
        betaln(x + alpha, total - x + beta)
        - betaln(alpha, beta)
        - betaln(x + 1, total - x + 1)
    )

    return -np.sum(log_prob)


def calculate_beta_binom_params(df: pd.DataFrame) -> Tuple[float, float]:
    df_threshold = df[df["n_starts"] >= MIN_STARTS_THRESHOLD]

    x = df_threshold["dnf"].values
    total = df_threshold["n_starts"].values

    result = minimize(
        log_likelihood,
        INITIAL_PARAMS,
        args=(x, total),
        method="L-BFGS-B",
        bounds=BOUNDS,
    )

    alpha0, beta0 = result.x

    return alpha0, beta0


def generate_beta_binomial_samples(
    n: int, alpha: float, beta: float, size: int
) -> np.ndarray:
    p = np.random.beta(alpha, beta, size)
    samples = np.random.binomial(n, p, size)

    return samples


def create_sample_plot(alpha0: float, beta0: float) -> None:
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
    plt.xlim(-0.5, N_TRIALS / 10 + 0.5)
    plt.xlabel(f"DNFs per {N_TRIALS} starts")
    plt.ylabel("Count of Trainers")
    plt.title("Simulated Number of DNFs")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("output/smoothing/beta_binom_sample.png")


def create_smoothed_scatter_plot(df: pd.DataFrame, alpha0: float, beta0: float) -> None:
    df = df.dropna(subset=["dnf_pct", "n_starts", "smoothed_dnf_pct"])
    plt.figure()
    plt.scatter(
        df["dnf_pct"],
        df["smoothed_dnf_pct"],
        c=np.log(df["n_starts"]),
        cmap="bwr",
        alpha=0.5,
    )
    plt.plot([0, 0.15], [0, 0.15], color="gray", linestyle="--", alpha=0.8)
    plt.plot(
        [0, 1],
        [(alpha0) / (alpha0 + beta0), (alpha0) / (alpha0 + beta0)],
        color="black",
        linestyle="--",
    )
    plt.xlim(-0.005, 0.15)
    plt.ylim(-0.005, 0.15)
    plt.xlabel("Observed DNF Rate")
    plt.ylabel("Smoothed DNF Rate")
    plt.title("Observed vs. Smoothed DNF Rate")
    plt.colorbar(label="Log Number of Starts")
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("output/smoothing/smoothed_dnf_scatter.png")


def create_ranking_plot(df: pd.DataFrame, alpha0: float, beta0: float) -> None:
    df = df.dropna(subset=["dnf_pct", "n_starts", "smoothed_dnf_pct"])

    df["low"] = beta.ppf(0.025, alpha0 + df["dnf"], beta0 + df["n_starts"] - df["dnf"])
    df["high"] = beta.ppf(0.975, alpha0 + df["dnf"], beta0 + df["n_starts"] - df["dnf"])

    if df.shape[0] > 10:
        print("Warning: Too many trainers to plot. Showing a random sample of 10.")
        df = df.sample(10)

    df = df.sort_values("smoothed_dnf_pct").reset_index(drop=True)

    plt.figure()
    plt.errorbar(
        df["smoothed_dnf_pct"],
        df["trainer_name"],
        xerr=[df["smoothed_dnf_pct"] - df["low"], df["high"] - df["smoothed_dnf_pct"]],
        fmt="o",
        color="blue",
        ecolor="black",
        capsize=3,
    )
    plt.axvline(x=alpha0 / (alpha0 + beta0), color="red", linestyle="--")
    plt.xlabel("Smoothed DNF Percentage, with 95% credible interval")
    plt.ylabel("Trainer")
    plt.title("Trainer Rankings by Smoothed DNF Percentage")
    plt.tight_layout()
    plt.grid(axis="both", linestyle="--", alpha=0.7)
    plt.savefig("output/smoothing/ranking_plot.png")
