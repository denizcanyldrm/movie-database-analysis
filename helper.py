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
    # INPUT
    #     df - a dataframe holding the col_name column
    #     bar_title - string the title of plot
    #     plot - bool providing whether or not you want a plot back
    #
    # OUTPUT
    #     df_col - a dataframe with the proportions of how many individuals
    #     Displays a plot of pretty things related to the col_name column.

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
    # INPUT:
    # df - the pandas dataframe you want to search
    # col1 - the column name you want to look through
    # col2 - the column you want to count values from
    # look_for - a list of strings you want to search for in each row of df[col]
    #
    # OUTPUT:
    # new_df - a dataframe of each look_for with the count of how often it shows up

    new_df = defaultdict(int)
    for val in look_for:
        for idx in range(df.shape[0]):
            if val in df[col1][idx]:
                new_df[val] += int(df[col2][idx])

    new_df = pd.DataFrame(pd.Series(new_df)).reset_index()
    new_df.columns = [col1, col2]
    new_df.sort_values('count', ascending=False, inplace=True)
    return new_df


# Drop columns with all NaN values
def drop_col_all_missing(df):
    return df.dropna(how='all', axis=1)


# Drop only rows with all missing values
def drop_row_all_missing(df):
    return df.dropna(axis=0, how='all')


def fit_data_linear(X_train, y_train):
    linear_model = LinearRegression(normalize=True)
    linear_model.fit(X_train, y_train)
    return linear_model


def fit_data_svm(X_train, y_train):
    svm_model = svm.SVR()
    svm_model.fit(X_train, y_train)
    return svm_model


def split_train_test(X, y, rand_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=rand_state)
    return X_train, X_test, y_train, y_test


def predict_get_score_data(X, y, model):
    y_preds = model.predict(X)
    r2_score_ = r2_score(y, y_preds)
    return r2_score_


def get_categorical_cols(df):
    return df.select_dtypes(include=['object']).columns


def get_numerical_cols(df):
    return df.select_dtypes(include=['float', 'int']).columns


def get_correlation_matrix(df):
    h_map = sns.heatmap(df.corr(), annot=True, fmt=".2f")
    return h_map


def get_top10_by_popularity(df):
    top_ten_1 = df[['original_title', 'popularity']]
    top_ten_1 = top_ten_1.sort_values(by=['popularity'], ascending=False)[:10]
    top_ten_1.columns = ['Original Title', 'Popularity']
    cm = sns.light_palette("purple", as_cmap=True)
    top_ten_1.style.background_gradient(cmap=cm)
    return


def preprocess_data(df):
    # shuffle dataset
    movies_df = df.sample(frac=1)

    # set feature columns
    features_df = movies_df[['budget', 'popularity', 'revenue', 'vote_count']]
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
    # transforms data into the range [0, 1]
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(df_scaled)
    return df_scaled
