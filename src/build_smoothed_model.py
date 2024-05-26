from smoothing.load_data import load_data
import smoothing.beta_binom as bb



if __name__ == '__main__':

    df = load_data()
    df = bb.preprocess_data(df)
    alpha0, beta0 = bb.calculate_beta_binom_params(df)
    bb.create_sample_plot(alpha0, beta0)
    df['smoothed_dnf_pct'] = (df['dnf'] + alpha0) / (df['n_starts'] + alpha0 + beta0)
    bb.create_smoothed_scatter_plot(df, alpha0, beta0)
    bb.create_ranking_plot(df, alpha0, beta0)