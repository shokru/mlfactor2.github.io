import numpy as np                 # Activate the number-managing library
import pandas as pd                # Activate the data science library
import matplotlib.pyplot as plt    # Activate the plotting library
import urllib.request
import zipfile

# Size factor: small vs large firms
def compute_size_portfolios(df):
    """Compute size-sorted portfolio returns."""
    df = df.copy()
    
    # Create large/small indicator based on median market cap each month
    df['large'] = df.groupby('date')['Size12m'].transform(
        lambda x: x > x.median()
    )
    
    # Extract year
    df['year'] = df['date'].dt.year
    
    # Compute average return by year and size group
    result = df.groupby(['year', 'large'])['R1M'].mean().reset_index()
    result['large'] = result['large'].map({True: 'Large', False: 'Small'})
    
    return result

def generate_data():
  data_raw=pd.read_csv('factors_map_all_US.csv')       # Load the data
  idx_date=data_raw.index[(data_raw['date'] > '1999-12-31') & (data_raw['date'] < '2025-06-01')].tolist() # creating and index to retrive the dates
  data_ml=data_raw.iloc[idx_date]                      # filtering the dataset according to date index
  data_ml = data_ml.drop(columns = 'Unnamed: 0')

  features=list(data_ml.iloc[:,3:125].columns)
  # Keep the feature's column names (hard-coded, beware!)
  features_short =["Div_yld", "EPS", "Size12m", "Mom_LT", "Ocf", "PB", "Vol_LT"]

  df_median=[]  #creating empty placeholder for temporary dataframe
  df=[]         #creating empty placeholder for temporary dataframe
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
  data_ml['date'] = pd.to_datetime(data_ml['date']) 

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
  returns = returns.reset_index(names="date")    
  return data_ml, features, features_short, returns, stock_ids, stock_ids_short    

def compute_correlations_with_label(data_ml, features_short, label='R1M'):
    import warnings
    
    # Select relevant columns
    cols = features_short + [label, 'date']
    df = data_ml[cols].copy()
    
    # Compute correlations grouped by date, suppressing warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        correlations = df.groupby('date').apply(
            lambda x: x[features_short].corrwith(x[label])
        ).reset_index()
    
    # Melt to long format for plotting
    corr_long = correlations.melt(
        id_vars='date', 
        var_name='Predictor', 
        value_name='correlation'
    )
    
    # Remove NaN correlations (from constant columns)
    corr_long = corr_long.dropna(subset=['correlation'])
    
    return corr_long

def plot_conditional_expectations(data_ml, features, label='R1M'):
    """
    Plot conditional expectations: average returns as smooth functions of features.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'Size12m': '#F87E1F', 'Vol_LT': '#0570EA'}
    labels = {'Size12m': 'Market Cap', 'Vol_LT': 'Volatility'}
    
    for feature in features:
        # Use lowess smoothing
        from statsmodels.nonparametric.smoothers_lowess import lowess
        
        # Sort by feature
        sorted_data = data_ml[[feature, label]].dropna().sort_values(feature)
        
        # Apply LOWESS smoothing
        smoothed = lowess(sorted_data[label], sorted_data[feature], frac=0.3)
        
        ax.plot(smoothed[:, 0], smoothed[:, 1], 
                color=colors.get(feature, 'gray'),
                label=labels.get(feature, feature),
                linewidth=2)
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel(label)
    ax.legend(title='Predictor')
    ax.set_aspect(25)
    plt.tight_layout()
    plt.show()      

def compute_autocorrelations(data_ml, features, id_col='fsym_id'):
    """
    Compute first-order autocorrelations for each stock and feature.
    """
    import warnings
    from statsmodels.tsa.stattools import acf
    
    autocorrs = []
    
    for fsym_id in data_ml[id_col].unique():
        stock_data = data_ml[data_ml[id_col] == fsym_id]
        
        for feature in features:
            series = stock_data[feature].dropna()
            if len(series) > 2:
                # Compute lag-1 autocorrelation, suppressing warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    acf_val = acf(series, nlags=1, fft=False)[1]
                if not np.isnan(acf_val):  # Only keep valid values
                    autocorrs.append({
                        'fsym_id': fsym_id,
                        'feature': feature,
                        'acf': acf_val
                    })
    
    return pd.DataFrame(autocorrs)

def get_significance(p):
    if p < 0.001:
        return ' (***)'
    elif p < 0.01:
        return ' (**)'
    elif p < 0.05:
        return ' (*)'
    else:
        return ''
 