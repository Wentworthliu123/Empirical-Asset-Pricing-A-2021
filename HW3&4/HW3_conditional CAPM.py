#!/usr/bin/env python
# coding: utf-8

# # Empirical Asset Pricing A 2021
# ## Homework 3&4: on empirical tests for conditional CAPM, return predictability
# **Xinyu Liu, INSEAD**
# 
# **02.02.2021**

# ## Overview
# 
# The goal of this exercise is to get a sense of the testing procedures in conditional CAPM, and check predictability of the market return.

# ## Preparation: Import packages and access data
# 

# In[1]:


import pandas_datareader.data as web  # module for reading datasets directly from the web
#pip install pandas-datareader (in case you haven't install this package)
from pandas_datareader.famafrench import get_available_datasets
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import statsmodels.api as sm
import scipy as sp
from dateutil.relativedelta import relativedelta
import datapungibea as dpb
import os
# print latex 
# from IPython.display import display, Math


# In[2]:


###########################
# Fama French Factor Grabber
###########################
#https://randlow.github.io/posts/finance-economics/pandas-datareader-KF/
#Please refer to this link if you have any further questions.

#You can extract all the available datasets from Ken French's website and find that there are 297 of them. We can opt to see all the datasets available.
datasets = get_available_datasets()
print('No. of datasets:{}'.format(len(datasets)))
#datasets # comment out if you want to see all the datasets


# In[7]:


###########################
#Customize your data selection
###########################
#It is important to check the description of the dataset we access by using the following codes 
sdate='1969-01-01'
edate='2018-12-31'
dir = os.path.realpath('.')


# #### For $M kt-Rf, SMB, HML$ Factors:

# In[4]:


Datatoread='F-F_Research_Data_Factors'
ds_factors = web.DataReader(Datatoread,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_factors.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_factors['DESCR']))
#From the printed information we know that we need to select the "0" name in the dictionary
#copy the right dict for later examination
dfFactor = ds_factors[0].copy()
# 0 for monthly data and 1 for yearly data
dfFactor.reset_index(inplace=True)

#Date format adjustment
# dfFactor['Date']=dfFactor['Date'].dt.strftime('%Y-%m')
dfFactor = dfFactor.set_index(['Date'])
dfFactor.index=dfFactor.index.to_timestamp()
# dfFactor['Date']=dfFactor['Date'].dt.to_timestamp(freq='M').dt.strftime('%Y-%m')
#Obtained object dtype
# dfFactor.index=pd.to_datetime(dfFactor.index)
#Obtained dt64, which is needed for the plotting

RF = dfFactor['RF']
# dfFactor=dfFactor.drop(columns = ['RF'])
# I check the scale of the data by printing out the head:
dfFactor.head()


# #### For 25 portfolios formed on size and book-to-market (5 x 5)

# In[16]:


# I searched for the exact name for this portfolio set by methods mentioned above
#It is important to check the description of the dataset we access by using the following codes 
Datatoread_PORT='25_Portfolios_5x5'
Datatoread_PORT='25_Portfolios_ME_BETA_5x5'
ds_PORT = web.DataReader(Datatoread_PORT,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_PORT.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_PORT['DESCR']))
#From the printed information we know that we need to select the "0" name in the dictionary
#copy the right dict for later examination
dfPORT = ds_PORT[0].copy()
dfPORT.reset_index(inplace=True)

dfPORT = dfPORT.set_index(['Date'])
# I check the scale of the data by printing out the head:
dfPORT.head()


# #### For monthly time-series of the default spread (”Baa - Aaa”)

# In[53]:


# from fredapi import Fred
# fred = Fred(api_key='867c31a2baca3a69effa928b9b294289')
# Aaa = fred.get_series_latest_release('AAA')
# Baa = fred.get_series_latest_release('BAA')
####
# The API above is not stable so I make a local copy and access them below
####
filename = os.path.join(dir, 'Data','AAA.csv')
Aaa = pd.read_csv(filename,index_col='DATE',parse_dates=True)
filename = os.path.join(dir, 'Data','BAA.csv')
Baa = pd.read_csv(filename,index_col='DATE',parse_dates=True)

Bond_spread = pd.DataFrame({'Aaa':Aaa.iloc[:,0].values,'Baa':Baa.iloc[:,0].values},index = Aaa.index)
Bond_spread = Bond_spread[(Bond_spread.index<=pd.to_datetime(edate)) & (Bond_spread.index>=pd.to_datetime(sdate))]
Bond_spread['Spread'] = Bond_spread['Baa']- Bond_spread['Aaa']
dfFactor = dfFactor.merge(Bond_spread[['Spread']], how='inner', left_index=True, right_index=True)


# In[56]:


def portfolio_plot(df, num_subplot, plot_name='testing' ,figsize=(8,8), cmap ='twilight'):
    n = num_subplot
    fig, axes = plt.subplots(n,1,figsize=figsize,sharex=True,sharey=True)
    years_fmt = mdates.DateFormatter('%Y')
    # fig.suptitle('Time series of relevant variables',fontsize=16)
    # Add an origin point at the top of the dataframe
    dfcopy = df.copy()
#     dfcopy.index = dfcopy.index.to_timestamp()
#     origin = dfcopy.index[0]-relativedelta(months=1)
#     dfcopy.loc[origin,:] = [1]*len(dfcopy.columns)
#     dfcopy=dfcopy.sort_index()

    dfFactor_cum = dfcopy
    for k,factortitle in enumerate(dfcopy.columns):
        if n==1:
            ax = axes
        else:
            ax = axes[k//n]
        ax.plot(dfFactor_cum.index,dfFactor_cum[factortitle], label='{}: {:.2f}'.format(factortitle, dfFactor_cum[factortitle].mean()))
        ax.xaxis.set_major_formatter(years_fmt)
        colormap = plt.cm.get_cmap(cmap)   
        colors = [colormap(i) for i in np.linspace(0.3, 0.5,len(ax.lines))]
        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])
        ax.legend(fontsize = 10,loc=2)
    fig.text(0.04, 0.5, 'Time series of ' +plot_name, va='center', ha='center',rotation='vertical',fontsize = 14)
    plt.savefig("Time series of "+plot_name)
    plt.show()
portfolio_plot(dfFactor[['Spread', 'RF']], 1, plot_name='Spread and RF' ,figsize=(8,4), cmap ='twilight')


# #### For monthly time-series of labor income growth (BEA)

# In[57]:


BEA_data = dpb.data('FDA2D756-CC0A-4AAA-A1D5-980FA23F31BB') #or data = dpb.data("API Key")
NIPA_cons=BEA_data.NIPA('T20600',frequency='M')
#Download annual consumption data on nondurable goods from Table 2.6. 
#on “Personal Income and Its Disposition, Monthly”
NIPA_cons.reset_index(inplace=True)
Compensation_data=NIPA_cons[NIPA_cons['LineDescription']=='-Compensation of employees']
Compensation_data = Compensation_data.T.iloc[4:,:]
Compensation_data.columns=['Compensation']
Compensation_data.index = pd.to_datetime(Compensation_data.index.values, format='%YM%m')
Compensation_data['Income Growth'] = (Compensation_data['Compensation']-Compensation_data['Compensation'].shift(1))/Compensation_data['Compensation'].shift(1)
# Convert strings to datetime 
Compensation_data = Compensation_data[(Compensation_data.index<=pd.to_datetime(edate)) & (Compensation_data.index>=pd.to_datetime(sdate))]
Compensation_data['Mkt-RF'] = dfFactor['Mkt-RF']/100
Compensation_data['Income Growth'] = Compensation_data['Income Growth']
labor_market = (Compensation_data[['Income Growth','Mkt-RF']]+1).astype('f').resample('Y').prod()-1
portfolio_plot(labor_market, 1, plot_name='Income Growth and Mkt-RF (monthly)' ,figsize=(8,4), cmap ='twilight')
dfFactor['Labor'] = Compensation_data['Income Growth'].astype('f')*100
# I don't know why but the api is not stable so I kept a copy of data 
# Compensation_data.to_pickle('compensation')
#or [All just for saving the intermediary data]
# Compensation_data.to_csv(os.path.join(dir, 'Data','Compensation.csv'))
# Compensation_data = pd.read_pickle('compensation')
# dfFactor.to_csv(os.path.join(dir, 'Data','dfFactor.csv'))


# ## Test functions 
# #### Define the function for conducting cross-sectional test, where the first stage is a time series regression

# In[10]:


# I can import directly the saved dfFactor
filename = os.path.join(dir, 'Data','dfFactor.csv')
dfFactor = pd.read_csv(filename,index_col='Date',parse_dates=True)


# In[17]:


def FamaMacbeth_Test(factor_matrix, test_assets, RF):
    try:
        test_assets.index = test_assets.index.to_timestamp()
    except Exception:
        pass
    # Step one, time series regression, obtain estimated beta for each portfolio
    X = sm.add_constant(factor_matrix)
    beta_matrix = pd.DataFrame()
    for i in range(len(test_assets.columns)):
        y= test_assets.iloc[:,i]-RF
        model = sm.OLS(y, X)
        results = model.fit()
        beta_i = pd.DataFrame(results.params[1:]).T
        beta_matrix= pd.concat([beta_matrix, beta_i])
    beta_matrix.index = test_assets.columns

    # Step two, cross sectional regression, obtain estimated intercept and factor risk premium period by period
    X = sm.add_constant(beta_matrix)
    premium_matrix = pd.DataFrame()
    rsquare_matrix = []
    for i in range(len(test_assets.index)):
        # Note to be consisitent we should still use the excess return
        y= test_assets.iloc[i,:]-RF[i]
        model = sm.OLS(y, X)
        results = model.fit()
        premium_i = pd.DataFrame(results.params).T
        premium_matrix= pd.concat([premium_matrix, premium_i])
        
        rsquare_matrix.append(results.rsquared_adj)
    premium_matrix.index = factor_matrix.index
    
    ## Key formula to calculate the statistics
    point_estimate = premium_matrix.mean()
    N = len(test_assets.index)
    std = premium_matrix.std()/np.sqrt(N)
    df = N-1
    significant_level = 0.975
    critical_value = sp.stats.t.ppf(significant_level, df)
    CI = [point_estimate-std*critical_value, point_estimate+std*critical_value]
    reports = pd.DataFrame(point_estimate).T
    reports = reports.rename(index={0:'FM coef'})
    reports.loc['t-stats',:]= reports.iloc[0,:]/std

    print(reports.round(2).to_latex())
    return beta_matrix, premium_matrix, point_estimate, rsquare_matrix


# In[18]:


beta_matrix, premium_matrix, point_estimate, rsquare_mean = FamaMacbeth_Test(dfFactor[['Mkt-RF', 'Spread','Labor']], dfPORT, RF)


# In[73]:


beta_matrix, premium_matrix, point_estimate, rsquare_mean = FamaMacbeth_Test(dfFactor[['Mkt-RF']], dfPORT, RF)


# In[286]:


# Sensitivity check for the parameters
cut = 240
beta_matrix, premium_matrix, point_estimate, rsquare_mean = FamaMacbeth_Test(dfFactor[['Mkt-RF', 'Spread','Labor']].iloc[:cut,:], dfPORT.iloc[:cut,:], RF[:cut])


# In[21]:


# Rolling average calcualtion for list data
numbers = rsquare_mean
window_size = 120
numbers_series = pd.Series(numbers)
windows = numbers_series.rolling(window_size)
moving_averages = windows.mean()
moving_averages_list = moving_averages.tolist()
without_nans = moving_averages_list[window_size - 1:]


# In[22]:


# plot time series of rolling average
fig, axes = plt.subplots(1,1,figsize=(8,4),sharex=True,sharey=True)
fig.text(0.04, 0.5, r'$R^2_{adj}$', va='center', ha='center',rotation='vertical',fontsize = 14)
colormap = plt.cm.get_cmap('twilight') 
axes.plot(dfPORT.index[window_size - 1:],without_nans,c=".3")
axes.axhline(y=np.mean(rsquare_mean),color='r', linestyle='--',label='Average '+r'$R^2_{adj}$'+': {}'.format(np.round(np.mean(rsquare_mean),2)))
axes.legend(fontsize = 14)
plt.plot()
plt.savefig('Rsquared')
plt.show()


# In[23]:


# Make the output table more readable
beta_matrix = beta_matrix.round(2)
for content in beta_matrix.T.index:
    print_report = pd.DataFrame(beta_matrix.T.loc[content,:].values.reshape(5,5),columns= ["BM" + str(i+1) for i in range(5)], index= ["ME" + str(i+1) for i in range(5)])
    print_report = pd.concat([print_report], axis=1, keys=[content])
    print(print_report.to_latex())


# In[24]:


# Process result from regressions to plot scatter plot
X = sm.add_constant(beta_matrix)
Estimated = X @ point_estimate
Realized = (dfPORT.sub(RF,axis = 'index')).mean()


# In[26]:


# Make the scatter plot 
fig, axes = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True)
fig.text(0.04, 0.5, 'Realized', va='center', ha='center',rotation='vertical',fontsize = 14)
fig.text(0.5,0.04, 'Estimated', va='center', ha='center',rotation='horizontal',fontsize = 14)
colormap = plt.cm.get_cmap('twilight') 
colors = [colormap(i) for i in np.linspace(0.1, 0.5,5)]
axes[0].plot([0.2, 1], [0.2, 1], ls="--", c=".3")
for i in range(0,5):
    axes[0].scatter(Estimated[i*5:(i+1)*5],Realized[i*5:(i+1)*5],c=colors[i],label = 'ME'+str(i+1), s=140)
axes[0].legend(fontsize = 14)
axes[1].plot([0.2, 1], [0.2, 1], ls="--", c=".3")
for i in range(0,5):
    axes[1].scatter(Estimated[i::5],Realized[i::5],c=colors[i],label = 'BM'+str(i+1), s=140)
axes[1].legend(fontsize = 14)
plt.plot()
plt.savefig('Scatter_mebetaCAPM')
plt.show()


# ### Return predictability test
# 1. Default spread
# 2. Short rate

# In[144]:


to_predict= dfFactor[['Mkt-RF']].rolling(12).sum().shift(-12)


# In[145]:


# Make the scatter plot 
import seaborn as sns
fig, axes = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True)
colormap = plt.cm.get_cmap('twilight') 
colors = [colormap(i) for i in np.linspace(0.3, 0.5,5)]
# axes[0].plot([0.2, 1], [0.2, 1], ls="--", c=".3")
for i, k in enumerate(dfFactor[['Spread','RF']].columns):
    print(i,k)
    sns.regplot(dfFactor[[k]],to_predict['Mkt-RF'],ax= axes[i])
    axes[i].set_xlabel(k, fontsize = 14)
    axes[i].set_ylabel('MK-RF', fontsize = 14)
# plt.plot()
plt.savefig('Return_predictability')
plt.show()


# In[150]:


# Output the regression test result in latex
beta_matrix = pd.DataFrame()
for i in range(len(dfFactor[['Spread','RF']].columns)):
    y = to_predict[:-12]
    X = sm.add_constant(dfFactor[['Spread','RF']].iloc[:-12,i])
    model = sm.OLS(y, X)
    results = model.fit()
    beta_i = pd.DataFrame(results.params[1:]).T
    beta_i= beta_i.rename(index= {0:'coef'})
    beta_matrix= pd.concat([beta_matrix, beta_i])
    t_i = pd.DataFrame(results.tvalues[1:]).T
    t_i= t_i.rename(index= {0:'t'})
    beta_matrix= pd.concat([beta_matrix, t_i])    
print(beta_matrix.round(2).to_latex())

