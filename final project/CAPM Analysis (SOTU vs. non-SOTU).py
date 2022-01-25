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

# separate return of stocks and the return of market
def separate_columns(df):
    rx = df.iloc[:, list(range(5, len(df.columns) - 1)) + [-1]]
    rx.iloc[:, :len(df.columns) - 6] = rx.iloc[:, :len(df.columns) - 6] * 100

    rmarket = df.iloc[:, [1, -1]]
    rmarket.iloc[:, 0] = rmarket.iloc[:, 0] * 100
    return rx, rmarket

# separate the return data to sotu and non-sotu
def separate_sotu(df_return, index_name):
    return_sotu = df_return.loc[df_return[index_name] == 1]
    return_sotu = return_sotu.iloc[:, :-1]
    return_nonsotu = df_return.loc[df_return[index_name] == 0]
    return_nonsotu = return_nonsotu.iloc[:, :-1]
    return return_sotu, return_nonsotu

# linear regression based on CAPM model
def capm_regression(rx, rmarket):
    betas = np.empty([1, len(rx.columns)])
    betarx = np.empty([1, len(rx.columns)])
    alphas = np.empty([1, len(rx.columns)])
    r_squared = np.empty([1, len(rx.columns)])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    for col in range(len(rx.columns)):
        x = rmarket.iloc[:, 0].values.reshape(-1, 1)
        x_imputed = imputer.fit_transform(x)
        y = rx.iloc[:, col].values.reshape(-1, 1)
        y_imputed = imputer.fit_transform(y)
        reg = LinearRegression().fit(x_imputed, y_imputed)
        betas[0, col] = reg.coef_
        betarx[0, col] = sum(rmarket.iloc[:, 0].values) * reg.coef_ / len(rmarket.iloc[:, 0].values)
        alphas[0, col] = reg.intercept_
        r_squared[0, col] = reg.score(x_imputed, y_imputed)
    return betas, betarx, alphas, r_squared


# get result from moments and regression results
def get_result(rx, rmarket):
    # separate return and market data by sotu and non-sotu
    rx_sotu, rx_nonsotu = separate_sotu(rx, 'index0')
    rmarket_sotu, rmarket_nonsotu = separate_sotu(rmarket, 'index0')

    # get the moments
    moments_sotu = get_moments(rx_sotu)
    moments_nonsotu = get_moments(rx_nonsotu)
    split = np.full((5, 1), '|')
    result = np.hstack([moments_sotu, split, moments_nonsotu])

    # run linear regressions
    betas_sotu, betarx_sotu, alphas_sotu, r_sotu = capm_regression(rx_sotu, rmarket_sotu)
    betas_nonsotu, betarx_nonsotu, alphas_nonsotu, r_nonsotu = capm_regression(rx_nonsotu, rmarket_nonsotu)

    betas = np.hstack([betas_sotu, [['|']], betas_nonsotu])
    betarx = np.hstack([betarx_sotu, [['|']], betarx_nonsotu])
    alphas = np.hstack([alphas_sotu, [['|']], alphas_nonsotu])

    result = np.vstack([result, betas, betarx, alphas])
    pd_result = pd.DataFrame(result)
    pd_result = pd_result.set_index([pd.Index(['mean', 'sd', 'skewness', 'kurtosis', 'obs',
                                               'beta', 'beta*E(rm)', 'alpha'])])
    pd_result.columns = list(rx_sotu.columns) + ['|'] + list(rx_nonsotu.columns)
    return pd_result

# the function for the whole analysis using Fama-French model
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
    # separate to rx and rmarket
    rx, rmarket = separate_columns(data_french)
    # get result in DataFrame
    pd_result = get_result(rx, rmarket)

    return pd_result

# transfer rows in results into lists
def result_to_list(df_result, index_name, num):
    list_sotu = df_result.loc[[index_name], :].values.flatten().tolist()[:num]
    list_sotu = [float(i) for i in list_sotu]
    list_nonsotu = df_result.loc[[index_name], :].values.flatten().tolist()[num * (-1):]
    list_nonsotu = [float(i) for i in list_nonsotu]
    return list_sotu, list_nonsotu


# plot results into graphs and save
def plot_graph(df_result, sname):
    plt.rcParams.update({'font.size': 24})
    num = int((len(df_result.columns) - 1) / 2)
    bar_width = 0.25
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(35, 14))

    # get the values for the graph
    mean_sotu, mean_nonsotu = result_to_list(df_result, 'mean', num)
    betarx_sotu, betarx_nonsotu = result_to_list(df_result, 'beta*E(rm)', num)
    alpha_sotu, alpha_nonsotu = result_to_list(df_result, 'alpha', num)

    # set positions for bars and labels
    br1 = np.arange(len(mean_sotu))
    br2 = [x + bar_width for x in br1]
    br3 = [x + bar_width for x in br2]
    xtick_labels = [tup[1] for tup in np.asarray(df_result.columns)[:num]]

    # plot ax1
    ax1.bar(br1, mean_sotu, color='azure', width=bar_width,
            edgecolor='grey', label='mean')
    ax1.bar(br2, betarx_sotu, color='darkseagreen', width=bar_width,
            edgecolor='grey', label='beta*E(rm)')
    ax1.bar(br3, alpha_sotu, color='lightsalmon', width=bar_width,
            edgecolor='grey', label='alpha')
    ax1.set_title('SOTU')
    ax1.set_xticks([r + bar_width for r in range(len(mean_sotu))])
    ax1.set_xticklabels(xtick_labels)
    ax1.legend()
    ax1.axhline(linewidth=1, color='black')

    # plot ax2
    ax2.bar(br1, mean_nonsotu, color='azure', width=bar_width,
            edgecolor='grey', label='mean')
    ax2.bar(br2, betarx_nonsotu, color='darkseagreen', width=bar_width,
            edgecolor='grey', label='beta*E(rm)')
    ax2.bar(br3, alpha_nonsotu, color='lightsalmon', width=bar_width,
            edgecolor='grey', label='alpha')
    ax2.set_title('non-SOTU')
    ax2.set_xticks([r + bar_width for r in range(len(mean_sotu))])
    ax2.set_xticklabels(xtick_labels)
    ax2.legend()
    ax2.axhline(linewidth=1, color='black')

    # plt.show()

    # save the figures
    save_name = os.path.join(PATH, 'Result', '{}.png'.format(sname))
    fig.savefig(save_name)

# get the date for sotu
sotu_date = read_excel('Data_Daily_1.1-2.4.xlsx', 'address', [0])

# List the Data_Daily excel files and their sheets that we are interested in.
# Please refer to the ReadMe sheet to check the details.
file_name = {'Data_Daily_1.1-2.4.xlsx': ['2.1.', '2.2.', '2.3.', '2.4.'],
             'Data_Daily_3.1-3.12.xlsx': ['3.1.', '3.4.', '3.7.'],
             'Data_Daily_4.1-4.12.xlsx': ['4.2.', '4.4.', '4.6.',
                                          '4.8.', '4.10.', '4.12.'],
             'Data_Daily_5.1-5.8.xlsx': ['5.1.', '5.2.', '5.3.']}

# start from 1933
tstart = 1937
# end to 2020 Feb
tend = 24683

# set file name and sheet name
for fname in file_name:
    sheet_list = file_name[fname]
    for sname in sheet_list:
        df_result = FF_analysis(tstart, tend, fname, sname, [0, 1])

        plot_graph(df_result, sname)