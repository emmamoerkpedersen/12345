
import os
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Set Working directory
datafolder=os.path.relpath(r'Data')



##########read rainfall files
#define fields in file and datatype for each field
headers = ['date','time', 'value']
dtypes = {'date': 'str','date': 'str', 'value': float}
file_names = ['CHHA.csv', 'DCCT.csv', 'TGLG.csv', 'WCHN.csv', 'NMKI.csv', 'MMMO.csv', 'SPPT.csv' ]

rainDict = {}
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

    # Save all rain data in dictionary
    rainDict[f'rain{file_number+1}'] = rain

####################################
##############read evaporation files

#define fields in file and datatype for each field
refet_names = ['351201.xlsx', '330201.xlsx', '328202.xlsx','328201.xlsx']
headers = ['date','value']
dtypes = {'date': 'str', 'value': float}

refetDict = {}

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

    refetDict[f'refet{refet_number+1}'] = refet


####################################
##############read flow series

## Y14 flow
areaY14 = 3791348074.963 #m²

headers = ['date','value']
dtypes = {'date': 'str', 'value': float}
#read file
flow=pd.read_csv(os.path.join(datafolder,'Y14-Q.txt'),sep=';',skiprows=3,header=None,names=headers,dtype=dtypes)
flow.iloc[:,0]=pd.to_datetime(flow.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
flow['date']=[dt.datetime(x.year,x.month,x.day) for x in flow['date']] #remove time information from the flow dates
flow.iloc[:,1]=flow.iloc[:,1]*86400.0 #convert m3/s to m3/d
flow.iloc[:,1]=flow.iloc[:,1]/areaY14*10**3 # convert to mm/d

## Y1C flow: 
areaY1C = 2149033230.112 #m²
#read file
flowY1C=pd.read_csv(os.path.join(datafolder,'Y1C-Q.txt'),sep=';',skiprows=3,header=None,names=headers,dtype=dtypes)
flowY1C.iloc[:,0]=pd.to_datetime(flowY1C.iloc[:,0],format='%Y-%m-%d %H:%M:%S')
flowY1C['date']=[dt.datetime(x.year,x.month,x.day) for x in flowY1C['date']] #remove time information from the flow dates
flowY1C.iloc[:,1]=flowY1C.iloc[:,1]*86400.0 #convert m3/s to m3/d
flowY1C.iloc[:,1]=flowY1C.iloc[:,1]/areaY1C*10**3 # convert to mm/d

####################################
#### Remake so all DataFrames have same time index (compared to rain)

startdate=rain['date'][np.min(np.where(np.logical_not(rain['value'].isna()))[0])]
enddate=rain['date'][rain.shape[0]-1]

#clip the refet and flow series to rain
for i, (key, values) in enumerate(refetDict.items()):
    refetDict[key] = refetDict[key].iloc[np.min(np.where(refetDict[key]['date']>=startdate)):np.max(np.where(refetDict[key]['date']<=enddate)),:]

flow=flow.iloc[np.min(np.where(flow['date']>=startdate)):np.max(np.where(flow['date']<=enddate)),:]
flowY1C=flowY1C.iloc[np.min(np.where(flowY1C['date']>=startdate)):np.max(np.where(flowY1C['date']<=enddate)),:]


####################################
# G-RUN DATA
# time series of average monthly runoff in mm/d
runoff_GRUN = pd.read_csv(os.path.join(datafolder,'runoffseries.csv'), usecols=['Block3-Lower','Block3-Y1C'] ,sep=';')

runoff_GRUN['Block3-Lower']=runoff_GRUN['Block3-Lower']*areaY14
runoff_GRUN['Block3-Y1C']=runoff_GRUN['Block3-Y1C']*areaY1C


fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(runoff_GRUN['Block3-Lower'], label= 'Block3-Lower G RUN data')
ax[1].plot(runoff_GRUN['Block3-Y1C'], label= 'Block3-Y1C G RUN data')
ax[0].legend(loc = 'upper right')
ax[1].legend(loc = 'upper right')
plt.show()


####################################
# Merge into one dataframe

from functools import reduce
# create a list of DataFrames from the dictionaries
refetPD = [pd.DataFrame(d) for d in refetDict.values()]
rainPD = [pd.DataFrame(d) for d in rainDict.values()]
# use reduce and pd.merge to merge the DataFrames into a single DataFrame
merged_refet = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), refetPD)
merged_rain = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), rainPD)

data_all = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), [merged_refet, merged_rain, flow, flowY1C])

# rename the columns to the keys of the original dictionaries
data_all.columns = ['date','PET_351201','PET_330201','PET_328202', 'PET_328201', 'rain_CHHA', 'rain_DCCT', 'rain_TGLG', 'rain_WCHN', 'rain_NMKI', 'rain_MMMO', 'rain_SPPT', 'flow', 'flowY1C']

data_all.set_index('date',inplace=True)
data_all.interpolate(method='linear',inplace=True)

#save dataframe as pickled file
data_all.to_pickle('dataframe.pkl')


####################################
# New dataframe with weighted average

#Calculate average
areasP = np.array([865216821,665778630,383284467.2,785339562.5, 131465617.03602804, 170280882.42145243, 150530436.137325912714005])

Pcatchment = pd.DataFrame((areasP[0]*data_all['rain_CHHA']+areasP[1]*data_all['rain_DCCT']+areasP[2]*data_all['rain_MMMO']+areasP[3]*data_all['rain_NMKI']+areasP[4]*data_all['rain_SPPT']+areasP[5]*data_all['rain_TGLG']+areasP[6]*data_all['rain_WCHN'])/np.sum(areasP), columns = ['value'])


areasPET = np.array([632073506, 443472572, 2017865143, 697936853 ])
PETcatchment = pd.DataFrame((areasPET[0]*data_all['PET_328201']+areasPET[1]*data_all['PET_328202']+areasPET[2]*data_all['PET_330201']+areasPET[3]*data_all['PET_351201'])/np.sum(areasPET), columns = ['value'])

# Create the new dataframe
data_frames2 = [Pcatchment, PETcatchment, flow, flowY1C]
data_Average = reduce(lambda  left,right: pd.merge(left,right,on='date',how='outer'), data_frames2)
data_Average.columns = ['date', 'Precipitation', 'PET', 'flow', 'flowY1C']
data_Average.set_index('date',inplace=True)

#fill remaining missing values by linear interpolation
#this requires that the data have been properly checked before and that there is no bigger gaps that need to be treated manually!
data_Average.interpolate(method='linear',inplace=True)

#save dataframe as pickled file
data_Average.to_pickle('dataframe2.pkl')


############### Rain + Flow plot
fig, axes = plt.subplots(nrows=len(rainDict)+1, ncols=1, sharex=True, figsize = (15,8))
for i, (key, values) in enumerate(rainDict.items()):
    ax = axes[i]
    ax.plot(values['date'], values['value']) #label = key
    # Shade areas for different data periods
    ax.axvspan(pd.to_datetime('2011-10-01'), pd.to_datetime('2012-03-29'), facecolor='red', alpha=0.2)  # Warmup period
    ax.axvspan(pd.to_datetime('2012-03-29'), pd.to_datetime('2019-02-15'), facecolor='blue', alpha=0.2)  # Train period
    ax.axvspan(pd.to_datetime('2019-02-15'), pd.to_datetime('2021-02-03'), facecolor='green', alpha=0.2)  # Validate period
    ax.axvspan(pd.to_datetime('2021-02-03'), pd.to_datetime('2022-01-01'), facecolor='orange', alpha=0.2)  # Test period

    #ax.legend(loc = 'upper right')

axes[len(rainDict)].plot(flow['date'], flow['value'], label = 'Flow')
axes[len(rainDict)].legend(loc = 'upper right')

fig.text(0.04, 0.5, 'P [mm/d]', ha='center', va='center', rotation='vertical')
plt.xlabel('Time [days]')
plt.show()