
import pandas as pd
import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":

    app.run(port=8080)
    dataset = pd.read_csv('./in/data2.csv')

    print(dataset)

    X = dataset.drop(['su'], axis=1)
    y = dataset[['su']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(4,16),
        'criterion' : ['mse', 'mae'],
        'max_depth' : range(2,11)
    }

    rand_est = RandomizedSearchCV(reg, parametros , n_iter=10, cv=3, scoring='neg_mean_absolute_error').fit(X,y)

    print(rand_est.best_estimator_)
    print(rand_est.best_params_)
    print(rand_est.predict(X.loc[[5]]))


    #implmentacion_randomizedSearchCV
    
