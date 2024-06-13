from utils.load_data import load_data
from utils.processing import build_features, group_trainer_info
import smoothing.beta_binom as bb

from utils.constants import OUTPUT_BUCKET
from smoothing.constants import COLS_FOR_SMOOTHING


def main():
    df = load_data(False, True)
    df = build_features(df)
    trainers = group_trainer_info(df)
    smoothing_params = bb.get_smoothing_params(trainers)
    trainers = bb.calculate_smoothed_rates(trainers, smoothing_p

    for c in COLS_FOR_SMOOTHING:
        bb.create_smoothed_scatter_plot(trainers, smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        bb.create_sample_plot(smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        bb.create_ranking_plot(trainers, smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        
    trainers.to_csv(f's3://{OUTPUT_BUCKET}/trainer_rates_smoothed.csv', index=False)


if __name__ == '__main__':

    main()
    