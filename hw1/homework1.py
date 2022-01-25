# Assignment 1
# Tianling Luo

import pandas as pd
import os
import matplotlib.pyplot as plt

PATH = r'D:\uchi\Fall\PPHA30536_Data and Programming for Public Policy II\homework-1-luotianling327'

def csv_to_df(fname, n):
    df = pd.read_csv(os.path.join(PATH, fname),
                           skiprows=4,
                           engine='c', )
    df.drop(df.tail(n).index, inplace=True)
    df = df.drop('GeoFips', axis=1)
    df = df.rename(columns={'GeoName': 'state'})
    return df

def merge_df(df_total, df_industry):
    df_total_reshaped = df_total.melt(id_vars=['state'],
                                      value_vars=['2000', '2017'],
                                      var_name='year',
                                      value_name='total')
    df_industry_reshaped = df_industry.melt(id_vars=['state', 'Description'],
                                            value_vars=['2000', '2017'],
                                            var_name='year')
    df_join = pd.merge(df_industry_reshaped, df_total_reshaped, on=['state', 'year'])
    df_join['value'] = df_join['value'].astype(float)
    df_join['share'] = df_join['value'] / df_join['total']
    df_join = df_join.drop(columns=['value', 'total'], axis=1)
    return df_join

# For Question 1
fname_total = 'SAEMP25N total.csv'
fname_industry = 'SAEMP25N by industry.csv'

df_total = csv_to_df(fname_total, 3)
df_industry = csv_to_df(fname_industry, 5)

df_industry = df_industry[df_industry['LineCode'].notnull()]
df_industry = df_industry.drop('LineCode', axis=1)
df_industry['2000'].replace({'(T)': float("NaN")}, inplace=True)
df_industry['2017'].replace({'(D)': float("NaN")}, inplace=True)

df_join = merge_df(df_total, df_industry)
df_join['Description'] = df_join['Description'].str.strip()
df_join_2 = df_join.set_index(['state', 'year', 'Description'])['share']
df_new = df_join_2.unstack()

df_new.to_csv(r'D:\uchi\Fall\PPHA30536_Data and Programming for Public Policy II\homework-1-luotianling327\data.csv')

# For Question 2(a)
df_new.reset_index(inplace=True)
df_manu_2000 = df_new.loc[df_new['year'] == '2000'][['state', 'year', 'Manufacturing']]
df_manu_2000_5 = df_manu_2000.sort_values(by=['Manufacturing'], ascending=False).head(5)
print(df_manu_2000_5)

# The Output is:
# Description           state  year  Manufacturing
# 28                  Indiana  2000       0.184840
# 98                Wisconsin  2000       0.177212
# 44                 Michigan  2000       0.162207
# 6                  Arkansas  2000       0.162043
# 66           North Carolina  2000       0.159088
# Hence, the states with the top five share of manufacturing employment in 2000
# are Indiana, Wisconsin, Michigan, Arkansas, and North Carolina.

manu_5 = ['Indiana', 'Wisconsin', 'Michigan', 'Arkansas', 'North Carolina']
df_manu_2017_5 = df_new.loc[df_new['year'] == '2017'][['state', 'year', 'Manufacturing']].loc[df_new['state'].isin(manu_5)]
df_manu_5 = pd.merge(df_manu_2000_5, df_manu_2017_5, on=['state'])

ax = plt.gca()
df_manu_5.plot(kind='line', x='state', y='Manufacturing_x', color='blue', ax=ax)
df_manu_5.plot(kind='line', x='state', y='Manufacturing_y', color='red', ax=ax)
ax.legend(['2000', '2017'])
plt.show()

# For Question 2(b)
df_new = df_new.set_index(['state'])
df_2000 = df_new.loc[df_new['year'] == '2000']
df_2017 = df_new.loc[df_new['year'] == '2017']

df_2000 = df_2000.drop(['year'], axis=1)
df_2017 = df_2017.drop(['year'], axis=1)

df_stack_2000 = df_2000.stack().sort_values(ascending = False).iloc[:5]
print(df_stack_2000)

# The states and industries having the top 5 highest concentration of employment in 2000 are:
# state                 Industry
# District of Columbia  Government and government enterprises    0.327406
# Alaska                Government and government enterprises    0.248069
# Hawaii                Government and government enterprises    0.221494
# New Mexico            Government and government enterprises    0.209680
# Wyoming               Government and government enterprises    0.199986

df_stack_2017 = df_2017.stack().sort_values(ascending = False).iloc[:5]
print(df_stack_2017)

# The states and industries having the top 5 highest concentration of employment in 2017 are:
# state                 Industry
# District of Columbia  Government and government enterprises    0.277521
# Alaska                Government and government enterprises    0.225462
# Hawaii                Government and government enterprises    0.197694
# New Mexico            Government and government enterprises    0.187749
# Wyoming               Government and government enterprises    0.187475