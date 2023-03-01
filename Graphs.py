#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LoadData import file_names

df=pd.read_pickle('dataframe.pkl')


####################
#plot time series
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})


fig, ax = plt.subplots(8, 1, sharex=True)
ax[0].plot(df['rain_CHHA'], label = file_names[0], color = 'lightgreen');ax[0].set_ylabel('P [mm/d]')
ax[1].plot(df['rain_DCCT'], label = file_names[1], color = 'limegreen');ax[1].set_ylabel('P [mm/d]')
ax[2].plot(df['rain_TGLG'], label = file_names[2], color = 'darkgreen');ax[2].set_ylabel('P [mm/d]')
ax[3].plot(df['rain_WCHN'], label = file_names[3], color = 'green');ax[3].set_ylabel('P [mm/d]')
ax[4].plot(df['rain_NMKI'], label = file_names[4], color = 'green');ax[4].set_ylabel('P [mm/d]')
ax[5].plot(df['rain_MMMO'], label = file_names[5], color = 'green');ax[5].set_ylabel('P [mm/d]')
ax[6].plot(df['rain_SPPT'], label = file_names[6], color = 'green');ax[6].set_ylabel('P [mm/d]')
ax[7].plot(df['flow'],label='Flow');ax[7].set_ylabel('Q [mm/d]')
plt.xlabel('Time [days]')
ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'upper right')
ax[2].legend(loc = 'upper right')
ax[3].legend(loc = 'upper right')
ax[4].legend(loc = 'upper right')
ax[5].legend(loc = 'upper right')
ax[6].legend(loc = 'upper right')
ax[7].legend(loc = 'upper right')
plt.show()
plt.savefig("C:/Users/magnu/OneDrive/Dokumenter/Kanddidat/2. Semester/Water modelling and observations/Github/12345")

#### SCATTERPLOTS

# Without altering the data
plt.scatter(df['rain_WCHN'],df['flow'])

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
plt.scatter(df_agg['rain_WCHN'],df_agg['flow'])
plt.show()

#extract values for summer months (april to november) only and plot
select=np.logical_and(np.array(df_agg.index.month)>4,np.array(df_agg.index.month)<11)
# Only summer months
plt.scatter(df_agg['rain_WCHN'][select],df_agg['flow'][select])
plt.show()
# Cross correlation

plt.xcorr(df['flow'], df['rain_WCHN'], maxlags = 20)
plt.show()
