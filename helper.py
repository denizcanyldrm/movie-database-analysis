import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from collections import defaultdict
import seaborn as sns


def clean_and_plot(df, col_name, bar_title, possible_vals, plot=True):
    """Displays a plot of pretty things related to the given column name of a dataframe

    Args:
        df (dataframe): a dataframe holding the col_name column
        col_name (str): one of the column name of df dataframe
        bar_title (str): string the title of plot
        possible_vals (list): list of possible values that col_name column has
        plot (bool): bool providing whether or not you want a plot back

    Returns:
        dataframe: a dataframe with the proportions of how many individuals
    """

    df_col = df[col_name].value_counts().reset_index()
    df_col.rename(columns={'index': col_name, col_name: 'count'}, inplace=True)
    df_col = total_count(df_col, col_name, 'count', possible_vals)
    df_col['count'] = df_col['count'] / df_col['count'].sum()

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.barh(df_col.iloc[:, 0], df_col.iloc[:, 1], color='purple')
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()
        ax.set_title(bar_title, loc='left', pad=10)

        for i in ax.patches:
            txt = str(round(round(i.get_width(), 4) * 100, 2)) + '%'
            ax.text(x=i.get_width(), y=i.get_y(), s=txt, horizontalalignment='left', weight='bold', verticalalignment='top')

        plt.show()

    return df_col


def total_count(df, col1, col2, look_for):
    """Calculates occurances of each element of a given list

    Args:
        df (dataframe): the pandas dataframe you want to search
        col1 (str): the column name you want to look through
        col2 (str): the column you want to count values from
        look_for (list): a list of strings you want to search for in each row of df[col]

    Returns:
        dataframe: a dataframe of each look_for with the count of how often it shows up
    """

    new_df = defaultdict(int)
    for val in look_for:
        for idx in range(df.shape[0]):
            if val in df[col1][idx]:
                new_df[val] += int(df[col2][idx])

    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df


def drop_col_all_missing(df):
    """Drops colunms with all missing values of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a dataframe having columns with all nan values are dropped
    """

    return df.dropna(how='all', axis=1)


def drop_row_all_missing(df):
    """Drops rows with all missing values of a dataframe

    Args:
    df (dataframe): a pandas dataframe

    Returns:
    dataframe: a dataframe having rows with all missing values are dropped
    """

    return df.dropna(axis=0, how='all')


def fit_data_linear(X_train, y_train):
    """Fits linear model to given args

    Args:
        X_train (dataframe): training data for linear model
        y_train (dataframe): training labels for linear model

    Returns:
        classifier: a linear regression model
    """

    linear_model = LinearRegression(normalize=True)
    linear_model.fit(X_train, y_train)
    return linear_model


def fit_data_svm(X_train, y_train):
    """Fits SVR model to given args

    Args:
        X_train (dataframe): training data for SVR model
        y_train (dataframe): training labels for SVR model

    Returns:
        classifier: a SVR model
    """

    svm_model = svm.SVR()
    svm_model.fit(X_train, y_train)
    return svm_model


def split_train_test(X, y, rand_state=42):
    """Splits X and y data into the train and test data

    Args:
        X (dataframe): all data
        y (dataframe): all labels
        rand_state (int): an integer for reproducible output across multiple function calls

    Returns:
        dataframe: a pandas dataframe training data
        dataframe: a pandas dataframe test data
        dataframe: a pandas dataframe training labels
        dataframe: a pandas dataframe test labels
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=rand_state)
    return X_train, X_test, y_train, y_test


def predict_get_score_data(X, y, model):
    """Given a model predicts X data and calculate r2 score

    Args:
        X (dataframe): values that we want to predict
        y (dataframe): true labels that corresponds to each X example
        model (classifier): a linear regression or SVR model

    Returns:
        float: r2 score of the model
    """

    y_preds = model.predict(X)
    r2_score_ = r2_score(y, y_preds)
    return r2_score_


def get_categorical_cols(df):
    """Gets categorical columns of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a dataframe with only categorical columns
    """

    return df.select_dtypes(include=['object']).columns


def get_numerical_cols(df):
    """Gets numerical columns of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a dataframe with only numerical columns
    """

    return df.select_dtypes(include=['float', 'int']).columns


def get_correlation_matrix(df):
    """Gets correlation matrix of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        heatmap: a heatmap that shows correlation matrix for df dataframe
    """

    h_map = sns.heatmap(df.corr(), annot=True, fmt=".2f")
    return h_map


def get_top10_by_popularity(df):
    """Plots top 10 movies with respect to popularity

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a pandas dataframe holds top 10 movies with respect to popularity
        list: a list of RGB tuples
    """

    top_ten_1 = df[['original_title', 'popularity']]
    top_ten_1 = top_ten_1.sort_values(by=['popularity'], ascending=False)[:10]
    top_ten_1.columns = ['Original Title', 'Popularity']
    cm = sns.light_palette("purple", as_cmap=True)

    return top_ten_1, cm


def preprocess_data(df):
    """Preprocesses the dataframe by applying data cleansing methods (Shuffling dataset, Changing infinity values to nan, Dropping columns and rows with all missing values, Filling mean for nan values in numerical columns, Dropping columns with all zeros)

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a pandas dataframe having only feature columns where some data cleansing methods applied
    """

    # shuffle dataset
    df = df.sample(frac=1)

    # set feature columns
    features_df = df[['budget', 'popularity', 'revenue', 'vote_count']]
    # change infinity values to nan
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    # drop columns with all missing
    features_df = drop_col_all_missing(features_df)
    # drop rows with all missing
    features_df = drop_row_all_missing(features_df)
    print(features_df.shape)

    # get numerical and categorical columns
    num_vars = get_numerical_cols(features_df)
    cat_cols = get_categorical_cols(features_df)
    print('num_var len: ' + str(len(num_vars)))
    print('cat_cols len: ' + str(len(cat_cols)))

    # fill mean for nan values in numerical values
    for col in num_vars:
        features_df[col].fillna(features_df[col].mean(), inplace=True)

    # drop columns with all zeros
    features_df = features_df.loc[:, (features_df != 0).any(axis=0)]
    print(features_df.shape)

    # drop rows with all zeros
    features_df = features_df.loc[(features_df != 0).any(axis=1)]
    print(features_df.shape)
    return features_df


def scale_data(df):
    """Scales dataframe into the range [0, 1]

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        dataframe: a pandas dataframe whose values scaled into [0, 1]
    """

    # transforms data into the range [0, 1]
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(df_scaled)
    return df_scaled


def read_data(file_name):
    """Reads filename

    Args:
        file_name (str): a filename where we get data

    Returns:
        dataframe: a pandas dataframe holding data
    """

    df = pd.read_csv(file_name)
    return df


def print_histogram(df):
    """Prints histogram of a dataframe

    Args:
        df (dataframe): a dataframe that we want to print histogram of it

    Returns:
        None
    """

    print(df.hist())
    return


def get_col_row_numbers(df):
    """Gets the numbers of row and column of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        int: number of rows of df dataframe
        int: number of columns of df dataframe
    """

    num_rows = df.shape[0]
    num_cols = df.shape[1]
    return num_rows, num_cols


def print_col_names(df):
    """Prints column names of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        None
    """

    print(df.columns)
    return


def print_save_vote_average(df, col_name):
    """Prints and saves histogram of given column name of a dataframe

    Args:
        df (dataframe): a dataframe
        col_name (str): column name of df dataframe

    Returns:
        None
    """

    ax_va = df[col_name].plot.hist(alpha=1, color='purple', figsize=(8, 4))
    fig_va = ax_va.get_figure()
    fig_va.savefig('hist_' + col_name + '.png')
    return


def get_keyword_counts(df):
    """Gets keyword count list of 'keywords' column of a dataframe

    Args:
        df (dataframe): a pandas dataframe

    Returns:
        list: list of keyword counts of df dataframe
    """

    keyword_counts_list = []
    for i in range(len(df['keywords'])):
        keyword_counts_list.append(df['keywords'][i].count('id'))

    return keyword_counts_list


def create_prediction_result_matrix(train_score_linear, train_score_svm, test_score_linear, test_score_svm):
    """Creates prediction results of given parameters

    Args:
        train_score_linear (float):  linear regression r2 score for train data
        train_score_svm (float): SVR r2 score for train data
        test_score_linear (float): linear regression r2 score for test data
        test_score_svm (float): SVR r2 score for test data

    Returns:
        dataframe: a pandas dataframe that holds prediction scores
    """

    index = ['Train Score', 'Test Score']
    pd.options.display.float_format = '{:,.2f}'.format
    result_matrix = pd.DataFrame([[train_score_linear, train_score_svm], [test_score_linear, test_score_svm]],
                                 columns=['LR', 'SVR'], index=index)
    return result_matrix


def get_description(df):
    """Get description of a dataframe

    Args:
        df (dataframe):  a pandas dataframe

    Returns:
        dataframe: summary statistics of the dataframe
    """

    description = df.describe()
    return description
