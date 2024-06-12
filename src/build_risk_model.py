from typing import Dict
import pandas as pd
import pickle
import boto3

from utils.constants import LOCAL_PATH, OUTPUT_BUCKET
from utils.load_data import load_data
from utils.processing import create_train_test_split
from trainer_risk.constants import DATES_USED
from trainer_risk.preprocessing import create_full_dataset
from models.model_builds import build_linear_regressor, build_xgb_regressor
from models.eval import evaluate_regression, get_feature_importance


def main(local: bool, feature_mode: str) -> Dict:
    df = load_data(local)
    df = create_full_dataset(df, DATES_USED, feature_mode)
    data = create_train_test_split(df, test_size=0.2, valid_size=0.1, split_column='trainer_id', target_column='target_dnf_smoothed')

    lin_reg_model = build_linear_regressor(data)
    xgb_model = build_xgb_regressor(data)

    metrics = []
    for model in [lin_reg_model, xgb_model]:
        for dset in ['train', 'test', 'valid']:
            X = data[f'X_{dset}']
            y = data[f'y_{dset}']
            y_pred = model.predict(X)
            data[f'X_{dset}']['y_pred'] = y_pred
            data[f'X_{dset}']['y_true'] = y

            metric = evaluate_regression(y, y_pred)
            metric['model'] = model.__class__.__name__
            metric['dataset'] = dset
            metric = {'model': metric.pop('model'), 'dataset': metric.pop('dataset'), **metric}
            metrics.append(metric)  

    metrics = pd.DataFrame(metrics)
    lin_reg_ft_importance = get_feature_importance(lin_reg_model, data['X_train'])
    xgb_ft_importance = get_feature_importance(xgb_model, data['X_train'])

    if local:    
        metrics.to_csv(f'{LOCAL_PATH}/risk_model_metrics.csv', index=False)
        lin_reg_ft_importance.to_csv(f'{LOCAL_PATH}/risk_lin_reg_ft_importance.csv')
        xgb_ft_importance.to_csv(f'{LOCAL_PATH}/risk_xgb_ft_importance.csv')
        with open(f'{LOCAL_PATH}/risk_lin_reg_model.pkl', 'wb') as f:
            pickle.dump(lin_reg_model, f)
        with open(f'{LOCAL_PATH}/risk_xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
    else:
        metrics.to_csv(f's3://{OUTPUT_BUCKET}/risk_model_metrics.csv', index=False)
        lin_reg_ft_importance.to_csv(f's3://{OUTPUT_BUCKET}/risk_lin_reg_ft_importance.csv')
        xgb_ft_importance.to_csv(f's3://{OUTPUT_BUCKET}/risk_xgb_ft_importance.csv')
        
        s3_resource = boto3.resource('s3')
        pickle_byte_obj = pickle.dumps(lin_reg_model)
        s3_resource.Object(OUTPUT_BUCKET,'risk_lin_reg_model.pkl').put(Body=pickle_byte_obj)

        pickle_byte_obj = pickle.dumps(xgb_model)
        s3_resource.Object(OUTPUT_BUCKET,'risk_xgb_model.pkl').put(Body=pickle_byte_obj)

    output = {
        'metrics': metrics,
        'log_reg_model': lin_reg_model,
        'xgb_model': xgb_model,
        'data': data,
        'lin_reg_ft_importance': lin_reg_ft_importance,
        'xgb_ft_importance': xgb_ft_importance
    }

    return output



if __name__ == '__main__':
    output = main(False, 'date')
