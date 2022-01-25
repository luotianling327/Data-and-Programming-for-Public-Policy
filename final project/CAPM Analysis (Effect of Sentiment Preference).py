import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt

PATH = r'D:\uchi\2021Fall\PPHA30536_Data and Programming for Public Policy II\final-project-tianling-luo'
pd.options.mode.chained_assignment = None

# read data from excel file
def read_excel(fname, sname, header):
    df = pd.read_excel(os.path.join(PATH, 'Data', fname), header=header, sheet_name=sname)
    return df

# calculate moments including mean, std, skewness, kurtosis
def get_moments(rx):
    nanmean = np.nanmean(rx, axis=0)
    nanstd = np.nanstd(rx, axis=0)
    skew1 = skew(rx)
    kurtosis1 = kurtosis(rx)
    length = np.full((1, rx.shape[1]), rx.shape[0])[0]
    moment = np.stack([nanmean, nanstd, skew1, kurtosis1, length], axis=0)
    return moment

# match dates
def match_dates(df, df_date):
    df['index0'] = 0
    for ind in df.index:
        for ind2 in df_date.index:
            if df.loc[ind, df.columns[0]] == df_date.loc[ind2, df_date.columns[0]]:
                df.loc[ind, 'index0'] = 1
    return df

# linear regression based on CAPM model
def capm_regression(rx, rmarket, sentiment=None):
    betas = np.empty([1, len(rx.columns)])
    betarx = np.empty([1, len(rx.columns)])
    alphas = np.empty([1, len(rx.columns)])
    betas_2 = np.empty([1, len(rx.columns)])
    betasent = np.empty([1, len(rx.columns)])
    R_squared = np.empty([1, len(rx.columns)])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x = rmarket.iloc[:, 0].values.reshape(-1, 1)
    x_imputed = imputer.fit_transform(x)
    for col in range(len(rx.columns)):
        y = rx.iloc[:, col].values.reshape(-1, 1)
        y_imputed = imputer.fit_transform(y)
        if sentiment is not None:
            s = sotu_sentiment.iloc[:, 0].values.reshape(-1, 1)
            s_imputed = imputer.fit_transform(s)
            X = np.empty((len(x_imputed), 2))
            for num in range(len(x_imputed)):
                X[num, 0] = x_imputed[num]
                X[num, 1] = s_imputed[num]
            reg = LinearRegression().fit(X, y_imputed)
            betas[0, col] = reg.coef_[0, 0]
            betarx[0, col] = sum(rmarket.iloc[:, 0].values) * reg.coef_[0, 0] / len(rmarket.iloc[:, 0].values)
            betas_2[0, col] = reg.coef_[0, 1]
            betasent[0, col] = sum(sentiment.iloc[:, 0].values) * reg.coef_[0, 1] / len(sentiment.iloc[:, 0].values)
            alphas[0, col] = reg.intercept_
            R_squared[0, col] = reg.score(X, y_imputed)
        else:
            reg = LinearRegression().fit(x_imputed, y_imputed)
            betas[0, col] = reg.coef_
            betarx[0, col] = sum(rmarket.iloc[:, 0].values) * reg.coef_ / len(rmarket.iloc[:, 0].values)
            alphas[0, col] = reg.intercept_
            R_squared[0, col] = reg.score(x_imputed, y_imputed)

    if sentiment is not None:
        return betas, betarx, betas_2, betasent, alphas, R_squared
    else:
        return betas, betarx, alphas, R_squared

# the function for analysis using Fama-French model
def FF_analysis(tstart, tend, fname, sname, header):
    # get stock French data
    data_french = read_excel(fname, sname, header)
    # change column names
    data_french = data_french.rename(columns={'Unnamed: 0_level_1': 'Date',
                                              'Unnamed: 0_level_0': '',
                                              'Unnamed: 1_level_0': '',
                                              'Unnamed: 2_level_0': '',
                                              'Unnamed: 3_level_0': '',
                                              'Unnamed: 4_level_0': '', })
    # restrain to a time period
    data_french = data_french.iloc[tstart:tend, :]
    # change the date to datetime format
    data_french[data_french.columns[0]] = pd.to_datetime(data_french.iloc[:, 0], format='%Y%m%d')
    data_french = data_french.reset_index(drop=True)
    # transfer return to excess return
    for col in range(5, len(data_french.columns)):
        data_french[data_french.columns[col]] = data_french[data_french.columns[col]] - data_french[
            data_french.columns[4]]

    # match dates
    data_french = match_dates(data_french, sotu_date)

    return data_french

# separate return of stocks and the return of market
def separate_columns(df):
    rx = df.iloc[:, list(range(5, len(df.columns) - 1)) + [-1]]
    rx.iloc[:, :len(df.columns) - 6] = rx.iloc[:,:len(df.columns) - 6] * 100

    rmarket = df.iloc[:, [1,-1]]
    rmarket.iloc[:, 0] = rmarket.iloc[:, 0] * 100
    return rx, rmarket

# separate the return data to sotu and non-sotu
def separate_sotu(df_return, index_name):
    return_sotu = df_return.loc[df_return[index_name] == 1]
    return_sotu = return_sotu.iloc[:, :-1]
    return_nonsotu = df_return.loc[df_return[index_name] == 0]
    return_nonsotu = return_nonsotu.iloc[:, :-1]
    return return_sotu, return_nonsotu

# get the regression result
def regression_result(rx_sotu, rmarket_sotu, moments_sotu, index_list, sotu_sentiment=None):
    reg = capm_regression(rx_sotu, rmarket_sotu, sotu_sentiment)
    result = np.vstack([moments_sotu] + list(reg))
    df_result = pd.DataFrame(result)
    df_result = df_result.set_index([pd.Index(index_list)])
    df_result.columns = list(rx_sotu.columns)
    df_result = df_result.T
    return df_result

# plot line graphs
def plot_double_lines(df_1, df_2, label_1, label_2, title, sname):
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots()
    ax.plot(range(len(list(df_1.index))), df_1['R_squared'], 'b-', label=label_1)
    ax.plot(range(len(list(df_2.index))), df_2['R_squared'], 'r--', label=label_2)
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_ylabel('R_squared')
    plt.xticks(range(len(list(df_1.index))), list(df_1.index))

    # plt.show()

    # save the figures
    save_name = os.path.join(PATH, 'Result', '{}_R_squared.png'.format(sname))
    fig.savefig(save_name)

"""
Compare the model of: 
"excess return = beta*E(rm-rf) + alpha"
and 
"excess return = beta*E(rm-rf) + beta2*sentiment_grade + alpha"
"""

# get the date for sotu
sotu_date = read_excel('Data_Daily_1.1-2.4.xlsx', 'address', [0])

# get sotu date and sentiments data
sotu_sentiment = read_excel('sentiment.xlsx', 'Sheet1', [0])
sotu_sentiment = sotu_sentiment[['sentiment', 'Date']]

# List the Data_Daily excel files and their sheets that we are interested in.
# Please refer to the README sheet in these files to check the details.
file_name = {'Data_Daily_1.1-2.4.xlsx': ['2.1.', '2.2.', '2.3.', '2.4.'],
             'Data_Daily_3.1-3.12.xlsx': ['3.1.', '3.4.', '3.7.'],
             'Data_Daily_4.1-4.12.xlsx': ['4.2.', '4.4.', '4.6.',
                                          '4.8.', '4.10.', '4.12.'],
             'Data_Daily_5.1-5.8.xlsx': ['5.1.', '5.2.', '5.3.']}

# start from 1933
tstart = 1937
# end to 2020 Feb
tend = 24683

# R_squared comparison between two models
for fname in file_name:
    sheet_list = file_name[fname]
    for sname in sheet_list:
        data_french = FF_analysis(tstart, tend, fname, sname, [0, 1])
        # separate to rx and rmarket
        rx, rmarket = separate_columns(data_french)
        # separate sotu and non-sotu
        rx_sotu, rx_nonsotu = separate_sotu(rx, 'index0')
        rmarket_sotu, rmarket_nonsotu = separate_sotu(rmarket, 'index0')
        # get the moments
        moments_sotu = get_moments(rx_sotu)

        # get sotu beta, betarx, alpha, R^2 (regression without sentiment)
        index_list1 = ['mean', 'sd', 'skewness', 'kurtosis', 'obs', 'beta', 'beta*E(rm)', 'alpha', 'R_squared']
        df_result1 = regression_result(rx_sotu, rmarket_sotu, moments_sotu, index_list1)

        # get sotu beta, betarx, beta2, beta2sent, alpha (regression with sentiment)
        index_list2 = ['mean', 'sd', 'skewness', 'kurtosis', 'obs', 'beta', 'beta*E(rm)',
                       'beta_2', 'beta2*sentiment', 'alpha', 'R_squared']
        df_result2 = regression_result(rx_sotu, rmarket_sotu, moments_sotu, index_list2, sotu_sentiment)

        # plot line graphs
        plot_double_lines(df_result1, df_result2, 'without sentiment', 'with sentiment',
                          'Comparisons between R_squared with and without sentiment\n', sname)