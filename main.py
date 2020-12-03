
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from utils import Utils
from models import Models

if __name__ == "__main__":

    utils = Utils()
    

    data = utils.load_from_csv('./in/data2.csv')
    X, y = utils.features_target(data, ['su'],['su'])

    print(data)

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }

    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)


    joblib.dump(rand_est, './models/rand_est.pkl')


