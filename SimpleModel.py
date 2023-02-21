import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from LoadData import train, test, validate
####################################
#read data from pickle file that is generated by the script A_ReadSeries.py

# Set Working directory
datafolder=os.path.relpath(r'Data')

data_in=pd.read_pickle('dataframe2.pkl')

####################################
print('make sure that only 1D numpy arrays are provided as input to the model functions. NOT pandas objects!!!')
def simple_model(par,pnames,train):
    import lib_model as lb
    #
    Smaxsoil=par[pnames.index('Smaxsoil')];msoil=par[pnames.index('msoil')];betasoil=par[pnames.index('betasoil')];S0soil=par[pnames.index('S0soil')]
    cf=par[pnames.index('cf')] #crop factor for regulating ET
    outflow_u,states_u=lb.unit_soil_zone_storage(cf*data_in['PET'].to_numpy(),data_in['Precipitation'].to_numpy(),[Smaxsoil,msoil,betasoil,S0soil],return_ET=False)
    #
    tp=par[pnames.index('tp')];k=par[pnames.index('k')]
    streamflow=lb.unit_hydrograph(outflow_u, [tp, k])
    #
    baseflow=par[pnames.index('baseflow')]
    #
    streamflow=streamflow+baseflow
    return(streamflow)

####################################
#define model parameters

p0={'Smaxsoil':100,'msoil':1,'betasoil':2,'cf':0.1,'baseflow':3,'S0soil':1,'tp':2,'k':10}
#convert dictionary to lists that are used as input to the model function
pnames=list(p0.keys())
p0=list(p0.values())

####################################
#simulate
streamflow=simple_model(p0,pnames,data_in)

####################################
#plot observed and simulated streamflow
#the observations are in m3/d, we convert to mm/d by dividing with the catchment area and multiplying with 1000
#do not expect the simulated hydrograph to look very good, the input data here don't really make sense

plt.plot(data_in['flow'].to_numpy(),label='Obs')
plt.plot(streamflow,label='Sim')
plt.legend()
