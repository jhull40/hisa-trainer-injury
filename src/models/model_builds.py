from typing import Dict
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from xgboost import XGBClassifier, XGBRegressor

from utils.constants import SEED



def build_linear_classifier(data: Dict) -> LogisticRegressionCV:
    
    X_train = data['X_train']
    y_train = data['y_train']

    model = LogisticRegressionCV(cv=5, random_state=SEED, max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    return model



def build_xgb_classifier(data: Dict, **kwargs) -> XGBClassifier:
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']


    model = XGBClassifier(random_state=SEED, n_jobs=-1, early_stopping_rounds=10, **kwargs)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return model


def build_linear_regressor(data: Dict) -> LassoCV:
    
    X_train = data['X_train']
    y_train = data['y_train']

    model = LassoCV(cv=5, random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    return model


def build_xgb_regressor(data: Dict, **kwargs) -> XGBRegressor:
        
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']

    model = XGBRegressor(random_state=SEED, n_jobs=-1, early_stopping_rounds=10, **kwargs)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    return model