
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LoadData import file_names

df=pd.read_pickle('dataframe2.pkl')

####################
#plot time series

plt.rcParams.update({'font.size': 7})

# Plot all the data together
fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, sharex=True)
for i, key in enumerate(df.columns):
    ax = axes[i]
    ax.plot(df[key], label = key)
    ax.legend(loc = 'upper right')
fig.text(0.04, 0.5, '[mm/d]', ha='center', va='center', rotation='vertical')
plt.xlabel('Time [days]')
plt.show()


#### SCATTERPLOTS
# Without altering the data
plt.scatter(df['Precipitation'],df['flow'])
plt.show()
#aggregate to monthly resolution
#create an index that for each day indicates whether it belongs to the first month, the second month, and so on
factor=30
aggr_index=pd.Series(range(int((df.shape[0]+1)/factor)))
aggr_index=aggr_index.repeat(factor)
aggr_index=aggr_index.reset_index(drop=True)
aggr_index.head()

#use this index in a grouping operation. average all values with the same index
df=df.iloc[0:aggr_index.shape[0],:]
df['aggr_ind']=aggr_index.values
df_agg=df.groupby(['aggr_ind']).mean()

#get a time index for the aggregated values - we use the last day for each 30 days that are being aggregated
timedf=pd.DataFrame(df.index)
timedf['aggr_ind']=aggr_index.values
timedf_agg=timedf.groupby(['aggr_ind']).tail(1)

#assign this new time index to the aggregated dataframe
df_agg=df_agg.set_index(timedf_agg['date'])
# Gathered in monthly resolution
plt.scatter(df_agg['Precipitation'],df_agg['flow'])
plt.show()

#extract values for summer months (april to november) only and plot
select=np.logical_and(np.array(df_agg.index.month)>4,np.array(df_agg.index.month)<11)
# Only summer months
plt.scatter(df_agg['Precipitation'][select],df_agg['flow'][select])
plt.show()
# Cross correlation

plt.xcorr(df['flow'], df['Precipitation'], maxlags = 20)
plt.show()
