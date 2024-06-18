import pandas as pd

from utils.load_data import load_data
from utils.processing import build_features, group_trainer_info
from baseline_model.inference import group_baseline_predictions
import smoothing.beta_binom as bb

from utils.constants import OUTPUT_BUCKET
from smoothing.constants import COLS_FOR_SMOOTHING
from baseline_model.constants import (
    DNF_RATIO,
    VET_SCRATCH_RATIO,
    BADLY_BEATEN_RATIO,
    BREAKDOWN_RATIO
)


def main(local: bool):
    df = load_data(local, True)
    df = build_features(df)
    trainers = group_trainer_info(df)
    smoothing_params = bb.get_smoothing_params(trainers)
    trainers = bb.calculate_smoothed_rates(trainers, smoothing_params)
    trainers = group_baseline_predictions(df, trainers)
    trainers['actual'] = trainers['dnf'] * DNF_RATIO + trainers['vet_scratched'] * VET_SCRATCH_RATIO + trainers['badly_beaten'] * BADLY_BEATEN_RATIO + trainers['breakdown'] * BREAKDOWN_RATIO

    for c in COLS_FOR_SMOOTHING:
        bb.create_smoothed_scatter_plot(trainers, smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        bb.create_sample_plot(smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        bb.create_ranking_plot(trainers, smoothing_params[c]['alpha0'], smoothing_params[c]['beta0'], c)
        
    trainers.to_csv(f's3://{OUTPUT_BUCKET}/trainer_rates_smoothed.csv', index=False)
    
    return trainers



def smoothed_inference(races: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = build_features(races)
    trainers = group_trainer_info(df)
    smoothing_params = bb.load_smoothing_params()
    trainers = bb.calculate_smoothed_rates(trainers, smoothing_params)
    trainers = group_baseline_predictions(df, trainers)
    trainers['actual'] = trainers['dnf'] * DNF_RATIO + trainers['vet_scratched'] * VET_SCRATCH_RATIO + trainers['badly_beaten'] * BADLY_BEATEN_RATIO + trainers['breakdown'] * BREAKDOWN_RATIO
    
    
    trainers['inj_above_avg'] = trainers['actual'] - trainers['baseline_prediction']
    trainers['inj_above_avg_per_start'] = trainers['inj_above_avg'] / trainers['n_starts']
    
    trainers.to_csv(f's3://{OUTPUT_BUCKET}/trainer_rates_smoothed_{filename}.csv', index=False)

    return trainers



if __name__ == '__main__':

    trainers = main(False)
    