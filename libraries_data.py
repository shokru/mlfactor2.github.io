import numpy as np                 # Activate the number-managing library
import pandas as pd                # Activate the data science library
import matplotlib.pyplot as plt    # Activate the plotting library
import urllib.request
import zipfile

data_raw=pd.read_csv('factors_map_all_US.csv')       # Load the data
idx_date=data_raw.index[(data_raw['date'] > '1999-12-31') & (data_raw['date'] < '2025-06-01')].tolist() # creating and index to retrive the dates
data_ml=data_raw.iloc[idx_date]                      # filtering the dataset according to date index
data_ml = data_ml.drop(columns = 'Unnamed: 0')

features=list(data_ml.iloc[:,3:125].columns)
# Keep the feature's column names (hard-coded, beware!)
features_short =["Div_yld", "EPS", "Size12m",
                 "Mom_LT", "Ocf", "PB", "Vol_LT"]

df_median=[]  #creating empty placeholder for temporary dataframe
df=[]         #creating empty placeholder for temporary dataframe
import numpy as np
df_median=data_ml[['date','R1M','R12M']].groupby(
    ['date']).median() # computings medians for both labels at each date
df_median.rename(
    columns={"R1M": "R1M_median",
             "R12M": "R12M_median"},inplace=True)
df = pd.merge(data_ml,df_median,how='left', on=['date'])
# join the dataframes
data_ml['R1M_C'] = np.where( # Create the categorical labels
    df['R1M'] > df['R1M_median'], 1.0, 0.0)
data_ml['R12M_C'] = np.where( # Create the categorical labels
    df['R12M'] > df['R12M_median'], 1.0, 0.0)

separation_date = "2017-01-15"
idx_train=data_ml.index[(data_ml['date']< separation_date)].tolist()
idx_test=data_ml.index[(data_ml['date']>= separation_date)].tolist()

stock_ids_short=[]   # empty placeholder for temporary dataframe
stock_days=[]        # empty placeholder for temporary dataframe
stock_ids=data_ml['fsym_id'].unique() # A list of all stock_ids
stock_days=data_ml[['date','fsym_id']].groupby(
    ['fsym_id']).count().reset_index() # compute nbr data points/stock
stock_ids_short=stock_days.loc[
    stock_days['date'] == (stock_days['date'].max())]
# Stocks with full data
stock_ids_short=stock_ids_short['fsym_id'].unique()
# in order to get a list
is_stock_ids_short=data_ml['fsym_id'].isin(stock_ids_short)
returns=data_ml[is_stock_ids_short].pivot(
    index='date',columns='fsym_id',values='R1M') # returns matrix