import numpy as np
import pandas as pd
import scipy as sp
from numpy.random import RandomState, SeedSequence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.constants._codata import val
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from lib.standards import *

# Gets int from get_seed()
ran_seed = get_seed()

# Path constants
LOG_DIR = f"results/pca_mul_linear_regression"  # the main path for your method output
MODEL_DIR = f"{LOG_DIR}/models/"  # the path for the specific models
RESULT_DIR = f"{LOG_DIR}/"  # the path for the result csv file

def gives_x_all_param_header():
    x = []
    for i in range(1, 24):
        x.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                  'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                  'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return x

def handle_model(target):

    x, y = df[gives_x_all_param_header()].values, df[target].values

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = get_train_test_split(x, y)

    # creating a regression model
    '''mul_reg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('mul_linear_reg', linear_model.LinearRegression())
    ])

    # Making list with all the steps
    list_param = []
    for i in range(50, 51, 1):
        list_param.append(i)

    param_grid = {'pca__n_components': list_param}

    # Grid search
    grid_search = GridSearchCV(mul_reg_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=4, verbose=3)

    # Fit the model
    results = grid_search.fit(x_train, y_train)

    print('Best score: ', results.best_score_)
    print('Best parameters: ', results.best_params_)'''

    best_model = Pipeline([
        ('scaler', StandardScaler()),
        #('pca', PCA(n_components=results.best_params_['pca__n_components'])),
        ('mul_linear_reg', linear_model.LinearRegression(n_jobs=4))
    ])

    # fitting the model
    best_model.fit(x_train, y_train)

    # making predictions
    predictions_train = best_model.predict(x_train)
    predictions_test = best_model.predict(x_test)

    train_evaluation = evaluate_model(y_train, predictions_train)
    test_evaluation = evaluate_model(y_test, predictions_test)

    create_and_save_graph(target, y_test, predictions_test, f'{MODEL_DIR}{target}/{target}-plot.png')

    return get_evaluation_results(train_evaluation, test_evaluation)


# importing data and converting to float32
df = pd.read_csv('./data.csv').astype(np.float32)
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
# , lcl_epsr, mcl_epsr, pcl_epsr, acl_k, lcl_k, mcl_k, pcl_k
# Print and save results
print(result.to_string())
save_csv(result, f"{RESULT_DIR}result.csv")