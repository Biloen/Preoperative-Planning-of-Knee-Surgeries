import csv
import random
import scipy as sp
import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import pandas as pd
from lib.standards import *
import pickle


LOG_DIR = f"results/random_forest"           # the main path for your method output
MODEL_DIR = f"{LOG_DIR}/models/"                # the path for the specific models
RESULT_DIR = f"{LOG_DIR}/"                      # the path for the result csv file

rand_seed = get_seed()


def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x


def handle_model(target):
    # Set up and normalise the input
    #predictors = list(set(list(data.columns)) - set(result_columns))
    #data[predictors] = data[predictors] / data[predictors].max()  #Not used in RF yet. May remove.

    # Set up the domain and label of the data set
    x, y = data[gives_x_all_param_header()].values, data[target].values

    x_train, x_test, y_train, y_test = get_train_test_split(x, y)
    # all code below can be abstracted away in multiple functions.

    RFRegressor = RFR()
    pipe = Pipeline([('scaler', StandardScaler()), ("RFRegressor", RFRegressor)])

    n_estimators = [int(x) for x in np.linspace(start=30, stop=150, num=int(120 / 5))]
    max_features = [float(x) for x in np.linspace(start=0.40, stop=1.0, num=50)]
    max_depth = [int(x) for x in np.linspace(start=10, stop=120, num=int(110 / 5))]
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=19)]
    min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=20, num=20)]

    random_grid = {'RFRegressor__n_estimators': scipy.stats.randint(low=1, high=150),
                   'RFRegressor__max_features': scipy.stats.uniform(0.01, 1.0),
                   'RFRegressor__max_depth': scipy.stats.randint(1, 100),
                   'RFRegressor__min_samples_split': scipy.stats.randint(1, 50),
                   'RFRegressor__min_samples_leaf': scipy.stats.randint(1, 50)}

    rf_randomSearch = RandomizedSearchCV(estimator=pipe, param_distributions=random_grid,
                                         scoring="neg_root_mean_squared_error", n_iter=60, #60
                                         cv=5, verbose=3, n_jobs=-1) #-1 #cv=5

    #cv=ShuffleSplit(n_splits=1, test_size=0.2)

    result = rf_randomSearch.fit(x_train, y_train)

    final_regressor = RFR(
        n_estimators=result.best_params_["RFRegressor__n_estimators"],
        max_features=result.best_params_['RFRegressor__max_features'],
        max_depth=result.best_params_['RFRegressor__max_depth'],
        min_samples_split=result.best_params_['RFRegressor__min_samples_split'],
        bootstrap=result.best_params_['RFRegressor__min_samples_leaf'],
        verbose=3, n_jobs=8
    )  # params in here

    final_pipe = Pipeline([('scaler', StandardScaler()), ("final_regressor", final_regressor)])
    final_pipe.fit(x_train, y_train)

    predict_test = final_pipe.predict(x_test)
    predict_train = final_pipe.predict(x_train)

    train_evaluation = evaluate_model(y_train, predict_train)
    test_evaluation = evaluate_model(y_test, predict_test)

    create_and_save_graph(target, y_test, predict_test, f"{MODEL_DIR}{target}/{target}-plot.png")

    save_hyperparameters(target, result.best_params_, -result.best_score_, f"{RESULT_DIR}hyperparams.csv")

    results_of_randomsearch_df = pd.DataFrame(result.cv_results_)
    results_of_randomsearch_df.to_csv(f'{MODEL_DIR}{target}/{target}_GridsearchCV_Results.csv', mode='a', header=True)

    # save gridsearch result object for further use!
    filehandle = open(f'{MODEL_DIR}{target}/{target}_cvRes.p', 'wb')
    pickle.dump(result.cv_results_, filehandle)

    return get_evaluation_results(train_evaluation, test_evaluation)

data = pd.read_csv('./data.csv', index_col=0).astype(np.float32)
result_columns = get_result_columns()

# Create, train and evaluate all eight models
acl_epsr = pd.DataFrame(handle_model("ACL_epsr"), index=["ACL_epsr"])
lcl_epsr = pd.DataFrame(handle_model("LCL_epsr"), index=["LCL_epsr"])
mcl_epsr = pd.DataFrame(handle_model("MCL_epsr"), index=["MCL_epsr"])
pcl_epsr = pd.DataFrame(handle_model("PCL_epsr"), index=["PCL_epsr"])
acl_k = pd.DataFrame(handle_model("ACL_k"), index=["ACL_k"])
lcl_k = pd.DataFrame(handle_model("LCL_k"), index=["LCL_k"])
mcl_k = pd.DataFrame(handle_model("MCL_k"), index=["MCL_k"])
pcl_k = pd.DataFrame(handle_model("PCL_k"), index=["PCL_k"])

# Concatenate intermediate results
result = pd.concat([acl_epsr, lcl_epsr, mcl_epsr, pcl_epsr, acl_k, lcl_k, mcl_k, pcl_k])

# Print and save results
print(result.to_string())
save_csv(result, f"{RESULT_DIR}result.csv")