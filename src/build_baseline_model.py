import pandas as pd
import numpy as np
import pickle
import boto3

from utils.load_data import load_data
from utils.processing import build_features, get_prev_race_features
from baseline_model.preprocessing import preprocess_data
from baseline_model.constants import (
    TARGET,
    DNF_RATIO,
    VET_SCRATCH_RATIO,
    BADLY_BEATEN_RATIO,
    BREAKDOWN_RATIO
)

from utils.constants import LOCAL_PATH, OUTPUT_BUCKET
from utils.processing import create_train_test_split
from models.model_builds import build_linear_classifier, build_xgb_classifier, build_linear_regressor, build_xgb_regressor
from models.eval import evaluate_classification, evaluate_regression, get_feature_importance


def main(local=False):
    df = load_data(local, True)
    df = build_features(df)
    df, cols_for_model = preprocess_data(df)
    df = get_prev_race_features(df)
    df['target'] = df['dnf'] * DNF_RATIO + df['vet_scratched'] * VET_SCRATCH_RATIO + df['badly_beaten'] * BADLY_BEATEN_RATIO + df['breakdown'] * BREAKDOWN_RATIO
    df['target_binary'] = np.clip(df['target'], 0, 1)
    
    data = create_train_test_split(df[cols_for_model + [TARGET]], test_size=0.2, valid_size=0.1, split_column='registration_number', target_column=TARGET)

    lin_model = build_linear_regressor(data)
    xgb_model = build_xgb_regressor(data)

    metrics = []
    for model in [lin_model, xgb_model]:
        for dset in ['train', 'test', 'valid']:
            X = data[f'X_{dset}']
            y = data[f'y_{dset}']
            y_pred = model.predict(X)                
            metric = evaluate_regression(y, y_pred)
            metric['model'] = model.__class__.__name__
            metric['dataset'] = dset
            metric = {'model': metric.pop('model'), 'dataset': metric.pop('dataset'), **metric}
            metrics.append(metric)  

    metrics = pd.DataFrame(metrics)

    if local:    
        metrics.to_csv(f'{LOCAL_PATH}/baseline_model_metrics.csv', index=False)
        with open(f'{LOCAL_PATH}/baseline_log_reg_model.pkl', 'wb') as f:
            pickle.dump(log_reg_model, f)
        with open(f'{LOCAL_PATH}/baseline_xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
    else:
        metrics.to_csv(f's3://{OUTPUT_BUCKET}/baseline_model_metrics.csv', index=False)
        
        s3_resource = boto3.resource('s3')
        pickle_byte_obj = pickle.dumps(lin_model)
        s3_resource.Object(OUTPUT_BUCKET,'baseline_lin_model.pkl').put(Body=pickle_byte_obj)

        pickle_byte_obj = pickle.dumps(xgb_model)
        s3_resource.Object(OUTPUT_BUCKET,'baseline_xgb_model.pkl').put(Body=pickle_byte_obj)

    output = {
        'metrics': metrics,
        'lin_model': lin_model,
        'xgb_model': xgb_model,
        'data': data
    }

    return output



if __name__ == '__main__':
    output = main()
