# # Empirical Asset Pricing A 2021
# ## Homework 2: on empirical tests for asset pricing models 
# **Xinyu Liu, INSEAD**
# 
# **20.01.2021**

# ## Overview
# 
# The goal of this exercise is to get familiar with the common practice used to test classical FF 3-factor asset pricing model. Both time series and cross-sectional tests are implemented.

# ## Preparation: Import packages and access data
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
# print latex 
# from IPython.display import display, Math

###########################
# Fama French Factor Grabber
###########################
#https://randlow.github.io/posts/finance-economics/pandas-datareader-KF/
#Please refer to this link if you have any further questions.

#You can extract all the available datasets from Ken French's website and find that there are 297 of them. We can opt to see all the datasets available.
datasets = get_available_datasets()
print('No. of datasets:{}'.format(len(datasets)))
#datasets # comment out if you want to see all the datasets

###########################
#Customize your data selection
###########################
#It is important to check the description of the dataset we access by using the following codes 
sdate='1997-03-01'
edate='2017-02-27'


# #### For $M kt-Rf, SMB, HML$ Factors:

Datatoread='F-F_Research_Data_Factors'
ds_factors = web.DataReader(Datatoread,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_factors.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_factors['DESCR']))
#From the printed information we know that we need to select the "0" name in the dictionary
#copy the right dict for later examination
dfFactor = ds_factors[0].copy()
dfFactor.reset_index(inplace=True)

#Date format adjustment
# dfFactor['Date']=dfFactor['Date'].dt.strftime('%Y-%m')
dfFactor = dfFactor.set_index(['Date'])
# dfFactor['Date']=dfFactor['Date'].dt.to_timestamp(freq='M').dt.strftime('%Y-%m')
#Obtained object dtype
# dfFactor.index=pd.to_datetime(dfFactor.index)
#Obtained dt64, which is needed for the plotting

RF = dfFactor['RF']
dfFactor=dfFactor.drop(columns = ['RF'])
# I check the scale of the data by printing out the head:
dfFactor.head()


# #### For 25 portfolios formed on size and book-to-market (5 x 5)

# I searched for the exact name for this portfolio set by methods mentioned above
#It is important to check the description of the dataset we access by using the following codes 
Datatoread_PORT='25_Portfolios_5x5'
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


# #### For 10 portfolios formed on momentum

Datatoread_MOM='10_Portfolios_Prior_12_2'
ds_MOM = web.DataReader(Datatoread_MOM,'famafrench',start=sdate,end=edate) # Taking [0] as extracting 1F-F-Research_Data_Factors_2x3')
print('\nKEYS\n{}'.format(ds_MOM.keys()))
print('DATASET DESCRIPTION \n {}'.format(ds_MOM['DESCR']))
dfMOM = ds_MOM[0].copy()
dfMOM.reset_index(inplace=True)

dfMOM = dfMOM.set_index(['Date'])
# I check the scale of the data by printing out the head:
dfMOM.head()


# ## Test functions 
# #### Define the function for conducting time series test

def Time_Series_Test(factor_matrix, test_assets, RF):
    X = sm.add_constant(factor_matrix)
    const_value = list()
    t_value = list()
    rsquared_adj_value = list()
    risidual_matrix = pd.DataFrame()
    # Loop to perform regression
    # Note that we should deduct RF from the portfolio return to get the excess return
    for i in range(len(test_assets.columns)):
        y= test_assets.iloc[:,i]-RF
        model = sm.OLS(y, X)
        results = model.fit()
        const_value.append(results.params[0])
        t_value.append(results.tvalues[0])
        rsquared_adj_value.append(results.rsquared_adj)
        if i == 0:
            risidual_matrix = pd.DataFrame(results.resid,columns=[i])
        else:
            risidual_matrix=risidual_matrix.join(pd.DataFrame(results.resid,columns=[i]))
    # convert result into dataframe
    ts_result = {'intercept': const_value, 't-stats': t_value, 'R^2-adj': rsquared_adj_value, 'test_assets_name': test_assets.columns}
    ts_result = pd.DataFrame.from_dict(ts_result, orient='index')
    ts_result.columns = ts_result.loc['test_assets_name',:]
    ts_result = ts_result.drop(['test_assets_name'])
    del ts_result.columns.name
    
    # Compute GRS test statistics
    T = len(test_assets.index)
    N = len(test_assets.columns)
    mu_mkt = factor_matrix['Mkt-RF'].mean()
    sigma_mkt = factor_matrix['Mkt-RF'].std()
    alpha = ts_result.T['intercept']
    GRS_sigma = (risidual_matrix.T @ risidual_matrix)/T
    GRS_sigma = np.matrix(GRS_sigma)
    GRS_sigma = np.linalg.inv(GRS_sigma)
    GRS_sigma_inv = pd.DataFrame(GRS_sigma)
    ## Key formula to calculate the statistics
    J_1 = (T-N-1)/N*(1+(mu_mkt/sigma_mkt)**2)**(-1)*np.dot(np.dot(alpha.T,GRS_sigma_inv),alpha)
    
    # Test procedure
    df1 = N
    df2 = T-N-1
    p_value = 1 - sp.stats.f.cdf(J_1, df1, df2)
    print('The GRS test statistic J_1 is {:2.2f}, and its p-value is {:2.6f}'.format(J_1, p_value))
    ts_result=ts_result.astype(float).round(2)
    print(ts_result.to_latex())
    return ts_result, np.round(J_1, 3), np.round(p_value, 3)

ts_result, J_1, p_value= Time_Series_Test(dfFactor, dfMOM, RF)
ts_result, J_1, p_value= Time_Series_Test(dfFactor, dfPORT, RF)

# Make the output table more readable
for content in ts_result.index:
    print_report = pd.DataFrame(ts_result.loc[content,:].values.reshape(5,5),columns= ["BM" + str(i+1) for i in range(5)], index= ["ME" + str(i+1) for i in range(5)])
    print_report = pd.concat([print_report], axis=1, keys=[content])
    print(print_report.to_latex())


# ## Plot for building up intuitions

###########################
#Plot out the graphs
###########################
#See this link for detailed guidance on date ticks
# https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/date.html
# I am troubled by adjusting the format and making subplots for the whole evening and it turns out that things can be simplified in the following way:
years_fmt = mdates.DateFormatter('%Y')
#This will be used as input to adjust the axis label to be in the unit of year
n = len(dfFactor.columns)
fig, axes = plt.subplots(n,1,figsize=(8,8),sharex=True,sharey=True)
#Using sharex help making the plot simple and easy to read
# Create fig and axes class so I can then process with them in the for loop.
# fig.suptitle('Time series of relevant variables',fontsize=16)
for k,factortitle in enumerate(dfFactor.columns):
    ax = axes[k]
#     ax.set_xticks(dfFactor.index)
    ax.plot(dfFactor.index.to_timestamp(),dfFactor[factortitle])
    ax.axhline(y=dfFactor[factortitle].mean(),color='r', linestyle='--',label='Average monthly return is {:.2f}'.format(dfFactor[factortitle].mean()))
    ax.xaxis.set_major_formatter(years_fmt)
    ax.set_ylabel(factortitle,fontsize = 14)
    ax.legend(fontsize = 14,loc=2)
plt.savefig("Time series of momnthly factor returns")
plt.show()

print(dfFactor.corr().round(2).to_latex())


# #### Define a function to plot aggregate gross returns of factor mimicking portfolios and testing portfolios

def portfolio_plot(df, num_subplot, plot_name='testing' ,figsize=(8,8), cmap ='twilight'):
    n = num_subplot
    fig, axes = plt.subplots(n,1,figsize=figsize,sharex=True,sharey=True)

    # fig.suptitle('Time series of relevant variables',fontsize=16)
    # Add an origin point at the top of the dataframe
    dfcopy = df.copy()
    dfcopy.index = dfcopy.index.to_timestamp()
    origin = dfcopy.index[0]-relativedelta(months=1)
    dfcopy.loc[origin,:] = [1]*len(dfcopy.columns)
    dfcopy=dfcopy.sort_index()

    dfFactor_cum = (dfcopy/100+1).cumprod()
    for k,factortitle in enumerate(dfcopy.columns):
        if n==1:
            ax = axes
        else:
            ax = axes[k//n]
        ax.plot(dfFactor_cum.index,dfFactor_cum[factortitle], label='{}: {:.2f}'.format(factortitle, dfFactor_cum[factortitle][-1]))
        ax.xaxis.set_major_formatter(years_fmt)
        colormap = plt.cm.get_cmap(cmap)   
        colors = [colormap(i) for i in np.linspace(0.1, 0.5,len(ax.lines))]
        for i,j in enumerate(ax.lines):
            j.set_color(colors[i])
        ax.legend(fontsize = 10,loc=2)
    fig.text(0.04, 0.5, 'Aggregate returns for ' +plot_name+' portfolios', va='center', ha='center',rotation='vertical',fontsize = 14)
    plt.savefig("Time series of "+plot_name)
    plt.show()


portfolio_plot(dfFactor, 1, plot_name='factor' ,figsize=(8,4), cmap ='twilight')

portfolio_plot(dfMOM, 1, plot_name='momentum' ,figsize=(8,4), cmap ='twilight')

portfolio_plot(dfPORT, 5, plot_name="size & b-to-m" ,figsize=(8,8), cmap ='twilight')


# #### Define the function for conducting cross-sectional test
def FamaMacbeth_Test(factor_matrix, test_assets, RF):
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
    for i in range(len(test_assets.index)):
        # Note to be consisitent we should still use the excess return
        y= test_assets.iloc[i,:]-RF[i]
        model = sm.OLS(y, X)
        results = model.fit()
        premium_i = pd.DataFrame(results.params).T
        premium_matrix= pd.concat([premium_matrix, premium_i])
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
    return premium_matrix, point_estimate, reports

premium_matrix, point_estimate, reports = FamaMacbeth_Test(dfFactor, dfPORT, RF)

