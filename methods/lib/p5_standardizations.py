import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# gives header names of the data as is read as a string array
def gives_header_array():
    header_array = ['0', 'ACL_k', 'ACL_epsr', 'PCL_k', 'PCL_epsr', 'MCL_k', 'MCL_epsr', 'LCL_k', 'LCL_epsr']
    for i in range(1, 24):
        header_array.extend(['trans_x_' + str(i), 'trans_y_' + str(i), 'trans_z_' + str(i), 'rot_z_' + str(i),
                             'rot_x_' + str(i), 'rot_y_' + str(i), 'F_x_' + str(i), 'F_y_' + str(i), 'F_z_' + str(i),
                             'M_x_' + str(i), 'M_y_' + str(i), 'M_z_' + str(i)])
    return header_array


# settings for print_a_graph
# Change dots in graph
MarkerSize = 0.1
DotColor = 'Blue'

# name: ACL_k;ACL_epsr;PCL_k;PCL_epsr;MCL_k;MCL_epsr;LCL_k;LCL_epsr;
# max : 19999.967223695727;0.2499893799857;35998.983920124774;0.0899953475651;16499.836678978456;0.1899978896023;11999.932531514498;0.2299997624033;
# min : 8.630605623066;-0.049990436682;0.9692844182828;-0.2099850768209;0.3093996201696;-0.1099737331755;83.5306473649589;-0.0352885538294;

# settings for the line in the graph for each method
# example
# line_settings = [[], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25], [0, 20000], [-0.25, 0.25]]
line_settings = [[0, 0], [0, 20000], [-0.05, 0.25], [0, 36000], [-0.21, 0.09], [0, 16500], [-0.11, 0.19], [83, 12000], [-0.0355, 0.23]]

# Constants for the x axis in the graph
# example
# x_axis_settings = [[], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5]]
x_axis_settings = [[0, 0], [0, 20000], [-0.05, 0.25], [0, 36000], [-0.21, 0.09], [0, 16500], [-0.11, 0.19], [83, 12000], [-0.0355, 0.23]]

# Constants for the y axis in the graph
# example
# y_axis_settings = [[], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5], [0, 30000], [-0.5, 0.5]]
y_axis_settings = [[0, 0], [0, 20000], [-0.05, 0.25], [0, 36000], [-0.21, 0.09], [0, 16500], [-0.11, 0.19], [83, 12000], [-0.0355, 0.23]]


# prints a graph with the given data for the given targetindex of header at the savestring variable
def print_a_graph(target_data, prediction_data, target_index, save_string):
    # set data
    header = gives_header_array()
    plt.figure()
    plt.scatter(target_data, prediction_data, color=DotColor, s=MarkerSize)
    plt.plot(line_settings[target_index], line_settings[target_index], color='red')

    # set names
    plt.title(f'{header[target_index]} model')
    plt.xlabel(f'Actual {header[target_index]} value')
    plt.ylabel(f'Predicted {header[target_index]} value')

    # set axies
    # plt.xlim(x_axis_settings[target_index][0], x_axis_settings[target_index][1])
    # plt.ylim(y_axis_settings[target_index][0], y_axis_settings[target_index][1])

    # saves/shows graph
    # plt.show()
    plt.savefig(save_string)
    plt.close()


# save the results of r2 rmse mae in a file at save_string
def save_results(target_index, target_data, predicted_data, save_string):
    header = gives_header_array()

    r2 = r2_score(target_data, predicted_data)
    rmse = mean_squared_error(target_data, predicted_data, squared=False)
    mae = mean_absolute_error(target_data, predicted_data)

    file = open(save_string, 'a')
    if target_index == 1:
        file.write('name;R2;RMSE;MAE\n')
    file.write(f'{header[target_index]};{r2};{rmse};{mae}\n')
    file.close()


# settings for train_val_test_split
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1
random_seed = 69


# returns the train val test split for all methods
def train_val_test_split(data, target_index):
    header = gives_header_array()
    x = data[header[9:285]]
    y = data[header[target_index]]

    # Train gets the train_ratio of the data set (train = 80%, test = 20% - af det fulde datasæt)
    x_train, x_test, y_train, y_test = sk_train_test_split(x, y, test_size=1 - train_ratio, random_state=random_seed)

    # Both Validation and Test get 50% each of the remainder (val = 10%, test = 10% - af det fulde datasæt)
    x_val, x_test, y_val, y_test = sk_train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + val_ratio), random_state=random_seed)

    return x_train, y_train, x_val, y_val, x_test, y_test
# example of how to use it
# x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(your_dataframe, 1)


def train_test_split(data, target_index):
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data, target_index)
    x_all, y_all = np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val))

    return x_all, y_all, x_test, y_test


def save_results_from_search(results, save_string):
    resdf = pd.DataFrame(results.cv_results_)
    resdf.to_csv(save_string, mode='a')


def print_predicted_data(target_data, prediction, save_string):
    print('print_predicted_data DOSENT WORK\n'
          'print_predicted_data DOSENT WORK\n'
          'print_predicted_data DOSENT WORK\n'
          'print_predicted_data DOSENT WORK\n'
          'print_predicted_data DOSENT WORK\n'
          'print_predicted_data DOSENT WORK\n')
    dataframe_target = pd.DataFrame(target_data)
    dataframe_prediction = pd.DataFrame(prediction, columns=['prediction'])
    print_it = pd.concat([dataframe_target, dataframe_prediction])
    print_it.to_csv(save_string, mode='a')


#
#
#TESTS
#
#


def test_train_val_test_split():
    print('testing train_val_test_split(data, target_index)')

    x_train1, y_train1, x_val1, y_val1, x_test1, y_test1 = train_val_test_split(df, 1)
    x_train2, y_train2, x_val2, y_val2, x_test2, y_test2 = train_val_test_split(df, 1)

    if x_train1.equals(x_train2) and y_train1.equals(y_train2) and x_val1.equals(x_val2) and y_val1.equals(y_val2) and x_test1.equals(x_test2) and y_test1.equals(y_test2):
        print('1 and 2 is the same nice')
    else:
        print('1 and 2 is not the same FUCK')


def max_min_of_each_column(data):
    header = gives_header_array()
    file = open('C:\\Users\\houga\\Desktop\\uni\\P5-Knee-Surgery\\Support_vector_regression\\max_and_min_in_data.csv', 'a')
    for i in range(1, 9):
        file.write(f'{header[i]};')
    file.write(f'\n')

    for i in range(1, 9):
        file.write(f'{df[header[i]].max()};')
    file.write(f'\n')

    for i in range(1, 9):
        file.write(f'{df[header[i]].min()};')
    file.write(f'\n')

    file.close()



if __name__ == '__main__':
    df = pd.read_csv('../data_processing/final_final_final.csv')
    # test_train_val_test_split()
    max_min_of_each_column(df)
