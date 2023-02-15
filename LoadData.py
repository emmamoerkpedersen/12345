

import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set Working directory
os.chdir('/Users/emmamork/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Observation Modeling and Management of Water Sytstems/Assigment')
datafolder=os.path.relpath(r'Data')

##########read rainfall files
#define fields in file and datatype for each field
headers = ['date','time', 'value']
dtypes = {'date': 'str','date': 'str', 'value': float}
#read file
file_names = ['CHHA.csv', 'DCCT.csv', 'TGLG.csv', 'WCHN.csv', 'NMKI.csv', 'MMMO.csv', 'SPPT.csv' ]

# Load all 5 Rain stations
for file_number, file_name in enumerate(file_names):
    rain = pd.read_csv(os.path.join(datafolder,file_name),sep=',',skiprows=1,header=None,names=headers,dtype=dtypes)
    datetimestring = rain.iloc[:, 0] + ' ' + rain.iloc[:, 1]
    rain.iloc[:, 0] = pd.to_datetime(datetimestring, format='%Y-%m-%d %H:%M:%S')
    rain['value'] = rain['value'].replace(-999, np.nan)
    rain = rain.iloc[np.min(np.where(rain['date'] >= dt.datetime(year=2011, month=9, day=29))):, :]
    rain['date'] = [dt.datetime(x.year, x.month, x.day) for x in rain['date']]
    rain['value'] = rain['value'].replace(np.nan, 0)
    rain = rain.groupby('date').sum()
    rain = rain.resample('D', origin='start').fillna(0)
    rain['date'] = rain.index
    rain=rain.reset_index(drop=True)

    variable_name = f"rain{file_number + 1}"
    globals()[variable_name] = rain



########read evaporation files
#define fields in file and datatype for each field
headers = ['date','value']
dtypes = {'date': 'str', 'value': float}

refet_names = ['351201.xlsx', '330201.xlsx', '328202.xlsx','328201.xlsx']

#read files

for refet_number, refet_name in enumerate(refet_names):
    refet=pd.read_excel(os.path.join(datafolder,refet_name),skiprows=1,header=None,names=headers,dtype=dtypes)

    refet.iloc[:,0]=pd.to_datetime(refet.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
    #convert from mm/step (with step=1month) to mm/day and the resample to days, assuming that each value corresponds to accumulated evaporation in previous month
    #calculate number of days for each month in the datafile
    steplength=(refet.date.diff().to_numpy()/1000000000/60/60/24).astype(np.float32)
    steplength[0]=31
    #divide monthly values by number of days to get evaporation per day
    refet['value']=refet['value']/steplength 
    refet.index=refet['date'] #resample operation below requires date index - create this index, then resample, then revert back to normal integer index
    #create a series of daily values (where the evaporation  for all days in the same month is the same)
    refet=refet.resample('D', origin='start').backfill()
    refet['date']=refet.index
    refet=refet.reset_index(drop=True)


    variable_refet = f'refet{refet_number + 1}'
    globals()[variable_refet] = refet

#########read flow series
#define fields in file and datatype for each field
area = 3791348073.82 #m2


headers = ['date','value']
dtypes = {'date': 'str', 'value': float}
#read file
flow=pd.read_csv(os.path.join(datafolder,'Y14-Q.txt'),sep=';',skiprows=3,header=None,names=headers,dtype=dtypes)
flow.iloc[:,0]=pd.to_datetime(flow.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
flow['date']=[dt.datetime(x.year,x.month,x.day) for x in flow['date']] #remove time information from the flow dates
flow.iloc[:,1]=flow.iloc[:,1]*86400.0 #convert m3/s to m3/d
flow.iloc[:,1]=flow.iloc[:,1]/area*10**3 # convert to mm/d


################ TEST HVOR DATA IKKE ER LAVET OM IFT. REGN
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(8, 1, sharex=True)
ax[0].plot(rain1['date'], rain1['value'], label = file_names[0], color = 'lightgreen');ax[0].set_ylabel('P [mm/d]')
ax[1].plot(rain2['date'], rain2['value'], label = file_names[1], color = 'forestgreen');ax[1].set_ylabel('P [mm/d]')
ax[2].plot(rain3['date'], rain3['value'], label = file_names[2], color = 'limegreen');ax[2].set_ylabel('P [mm/d]')
ax[3].plot(rain4['date'], rain4['value'], label = file_names[3], color = 'darkgreen');ax[3].set_ylabel('P [mm/d]')
ax[4].plot(rain5['date'], rain5['value'], label = file_names[4], color = 'darkgreen');ax[4].set_ylabel('P [mm/d]')
ax[5].plot(rain6['date'], rain6['value'], label = file_names[5], color = 'darkgreen');ax[5].set_ylabel('P [mm/d]')
ax[6].plot(rain7['date'], rain7['value'], label = file_names[6], color = 'darkgreen');ax[6].set_ylabel('P [mm/d]')
ax[7].plot(flow['date'], flow['value'],label='Flow');ax[7].set_ylabel('Q [mm/d]')
plt.xlabel('Time [days]')
ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'upper right')
ax[2].legend(loc = 'upper right')
ax[3].legend(loc = 'upper right')
ax[4].legend(loc = 'upper right')
ax[5].legend(loc = 'upper right')
ax[6].legend(loc = 'upper right')
ax[7].legend(loc = 'upper right')







#### Remake so all DataFrames have same time index (compared to rain)

startdate=rain['date'][np.min(np.where(np.logical_not(rain['value'].isna()))[0])]
enddate=rain['date'][rain.shape[0]-1]
#clip the refet and flow series
refet1=refet1.iloc[np.min(np.where(refet1['date']>=startdate)):np.max(np.where(refet1['date']<=enddate)),:]
refet2=refet2.iloc[np.min(np.where(refet2['date']>=startdate)):np.max(np.where(refet2['date']<=enddate)),:]
refet3=refet3.iloc[np.min(np.where(refet3['date']>=startdate)):np.max(np.where(refet3['date']<=enddate)),:]
refet4=refet4.iloc[np.min(np.where(refet4['date']>=startdate)):np.max(np.where(refet4['date']<=enddate)),:]

flow=flow.iloc[np.min(np.where(flow['date']>=startdate)):np.max(np.where(flow['date']<=enddate)),:]


#Combine the 3 series into one dataframe with the same time index
#rain series is the shortest series, find dates where it starts and ends

#####################

from functools import reduce
#combine textfiles into one dataframes
data_frames = [refet1, refet2, refet3, refet4, rain1, rain2, rain3, rain4, rain5, rain6, rain7, flow ]

data_all = reduce(lambda  left,right: pd.merge(left,right,on='date',how='outer'), data_frames)
data_all.columns = ['date','PET_351201','PET_330201','PET_328202', 'PET_328201', 'rain_CHHA', 'rain_DCCT', 'rain_TGLG', 'rain_WCHN', 'rain_NMKI', 'rain_MMMO', 'rain_SPPT', 'flow']


data_all.set_index('date',inplace=True)
#####################
#fill remaining missing values by linear interpolation
#this requires that the data have been properly checked before and that there is no bigger gaps that need to be treated manually!
data_all.interpolate(method='linear',inplace=True)
#####################
#save dataframe as pickled file
data_all.to_pickle('dataframe.pkl')



