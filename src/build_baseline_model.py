from baseline_model.load_data import load_data
from baseline_model.preprocessing import preprocess_data, create_train_test_split
from models.build_logistic_regression import build_logistic_regression, evaluate_classification


def main():
    df = load_data()
    df = preprocess_data(df)
    data = create_train_test_split(df, test_size=0.2, valid_size=0.1)

    model = build_logistic_regression(data)
    for name in ['train', 'test', 'valid']:
        X = data[f'X_{name}']
        y = data[f'y_{name}']
        y_pred = model.predict(X)
        y_pred_soft_labels = model.predict_proba(X)[:, 1]

        metrics = evaluate_classification(name, y, y_pred, y_pred_soft_labels)
        print(f'{name}: {metrics}')




if __name__ == '__main__':
    main()