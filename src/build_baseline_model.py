import pandas as pd
import pickle

from baseline_model.load_data import load_data
from baseline_model.preprocessing import preprocess_data, create_train_test_split
from models.model_builds import build_linear_classifier, build_xgb_classifier
from models.eval import evaluate_classification


def main():
    df = load_data()
    df = preprocess_data(df)
    data = create_train_test_split(df, test_size=0.2, valid_size=0.1)

    log_reg_model = build_linear_classifier(data)
    xgb_model = build_xgb_classifier(data)

    metrics = []
    for model in [log_reg_model, xgb_model]:
        for dset in ['train', 'test', 'valid']:
            X = data[f'X_{dset}']
            y = data[f'y_{dset}']
            y_pred = model.predict(X)
            y_pred_soft_labels = model.predict_proba(X)[:, 1]

            metric = evaluate_classification(y, y_pred, y_pred_soft_labels)
            metric['model'] = model.__class__.__name__
            metric['dataset'] = dset
            metric = {'model': metric.pop('model'), 'dataset': metric.pop('dataset'), **metric}
            metrics.append(metric)  

    pd.DataFrame(metrics).to_csv('output/models/baseline_model_metrics.csv', index=False)

    with open('output/models/baseline_log_reg_model.pkl', 'wb') as f:
        pickle.dump(log_reg_model, f)

    with open('output/models/baseline_xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)


if __name__ == '__main__':
    main()
