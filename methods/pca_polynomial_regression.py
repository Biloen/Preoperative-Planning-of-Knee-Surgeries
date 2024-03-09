import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from lib.standards import *
import pickle

# Gets int from
ran_seed = get_seed()

# Path constants
LOG_DIR = f"results/pca_polynomial_regression"  # the main path for your method output
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
    # Set data up
    x = df[gives_x_all_param_header()].values
    y = df[target].values

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = get_train_test_split(x, y)

    # creating polynomial features
    poly_reg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('pca_poly_reg', PolynomialFeatures(degree=2, include_bias=False)),
        ('lin_reg', LinearRegression(n_jobs=4))
    ])

    # Making list with all the steps
    list_param = []
    for i in range(60, 101, 1): #range(60, 101, 1)
        list_param.append(i)

    param_grid = {'pca__n_components': list_param}

    # Grid search
    grid_search = GridSearchCV(poly_reg_model, param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=4, verbose=3)

    # Fit the model
    results = grid_search.fit(x_train, y_train)

    print('Best score: ', results.best_score_)
    print('Best parameters: ', results.best_params_)

    # Train the model
    best_poly_reg_model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=results.best_params_['pca__n_components'])),
        ('pca_poly_reg', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ('lin_reg', LinearRegression(n_jobs=4))
    ])

    # Fitting the model
    best_poly_reg_model.fit(x_train, y_train)

    # Making predictions
    predictions_train = best_poly_reg_model.predict(x_train)
    predictions_test = best_poly_reg_model.predict(x_test)

    train_evaluation = evaluate_model(y_train, predictions_train)
    test_evaluation = evaluate_model(y_test, predictions_test)

    create_and_save_graph(target, y_test, predictions_test, f'{MODEL_DIR}{target}/{target}-plot.png')

    save_hyperparameters(target, results.best_params_, -results.best_score_, f"{RESULT_DIR}hyperparams.csv")
    
    # saves the results of the gridsearch
    results_of_gridsearch_df = pd.DataFrame(results.cv_results_)
    results_of_gridsearch_df.to_csv(f'{MODEL_DIR}{target}/{target}_GridsearchCV_Results.csv', mode='a', header=True)

    #save gridsearch result object for further use!
    filehandle = open(f'{MODEL_DIR}{target}/{target}_cvRes.p', 'wb')
    pickle.dump(results.cv_results_, filehandle)

    return get_evaluation_results(train_evaluation, test_evaluation)


# importing data and converting it to float32
df = pd.read_csv('./data.csv', index_col=0).astype(np.float32)
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
