import json

from baseline_model.load_data import load_data
import smoothing.beta_binom as bb


def main():
    df = load_data(local=True)
    df = bb.preprocess_data(df)
    alpha, beta = bb.calculate_beta_binom_params(df)
    bb.create_sample_plot(alpha, beta)
    df['smoothed_dnf_pct'] = (df['dnf'] + alpha) / (df['n_starts'] + alpha + beta)
    df['smoothed_dnf_percentile'] = round(100*df['smoothed_dnf_pct'].rank(pct=True, ascending=False))
    bb.create_smoothed_scatter_plot(df, alpha, beta)
    bb.create_ranking_plot(df, alpha, beta)

    with open('output/smoothing/beta_binom_params.json', 'w') as f:
        json.dump({'alpha': alpha, 'beta': beta}, f)


if __name__ == '__main__':

    main()
    