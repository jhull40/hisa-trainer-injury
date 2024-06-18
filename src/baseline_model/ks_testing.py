import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from utils.processing import build_features, get_prev_race_features
from utils.load_data import load_data
from baseline_model.preprocessing import preprocess_data
from smoothing.constants import COLS_FOR_SMOOTHING


def main(local: bool) -> pd.DataFrame:
    df = load_data(local, True)

    df = build_features(df)
    df, cols_for_model = preprocess_data(df)
    df = get_prev_race_features(df)
    neg = df[(df['previous_race_dnf'] == 0) & (df['previous_race_scratched'] == 0) & (df['previous_race_badly_beaten'] == 0)]
    neg_vals = [v for v in neg['days_since_last_race'] if (v < 365 and v > 0)]
    ks_stats = []
    for c in COLS_FOR_SMOOTHING:
        pos = df[df[f'previous_race_{c}'] == 1]
        plt.figure()
        pos_vals = [v for v in pos['days_since_last_race'] if (v < 365 and v > 0)]
        ks = ks_2samp(neg_vals, pos_vals, alternative='greater')
        ks_stats.append({
            'value': c,
            'statistic': ks[0],
            'p_value': ks[1],
            'statistic_location': ks.statistic_location,
            'statistic_sign': ks.statistic_sign,
            'neg_vals_mean': np.mean(neg_vals),
            'pos_vals_mean': np.mean(pos_vals),
            'neg_vals_median': np.median(neg_vals),
            'pos_vals_median': np.median(pos_vals),
            'ratio': np.median(pos_vals) / np.median(neg_vals)
        })

        plt.hist(neg_vals, color='blue', alpha=0.5, density=True, bins=30)
        plt.hist(pos_vals, color='red', alpha=0.5, density=True, bins=30)
        plt.title(c)
        plt.savefig(f'output/models/{c}_days_to_next_race.png')


    ks_df = pd.DataFrame(ks_stats)
    ks_df.to_csv('output/models/ks_stats.csv', index=False)

    return ks_df