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
#print('make sure that only 1D numpy arrays are provided as input to the model functions. NOT pandas objects!!!')
#Inflow upper zone
def simple_model(par,pnames,data_input):
    import lib_model as lb
    # Soil storage
    Smaxsoil=par[pnames.index('Smaxsoil')];msoil=par[pnames.index('msoil')];betasoil=par[pnames.index('betasoil')];S0soil=par[pnames.index('S0soil')]
    cf=par[pnames.index('cf')] #crop factor for regulating ET
    outflow_u, states_u = lb.unit_soil_zone_storage(cf*data_input['PET'].to_numpy(),data_input['Precipitation'].to_numpy(),[Smaxsoil,msoil,betasoil,S0soil],return_ET=False)
    # Flow from other Y1C
    Smax_Y1C=par[pnames.index('Smax_Y1C')];PERC=par[pnames.index('PERC')];k0=par[pnames.index('k0')]; k1=par[pnames.index('k1')];S0_s=par[pnames.index('S0_s')]
    outflow_Y1C, percolation ,states_Y1C = lb.unit_hbv_shallow_storage(data_input['flowY1C'].to_numpy(),[Smax_Y1C,PERC,k0,k1,S0_s])
    #####################
    #add shallow storage
    Smax_s=par[pnames.index('Smax_s')]
    outflow_s, percolation, states_u = lb.unit_hbv_shallow_storage(outflow_u,[Smax_s,PERC,k0,k1,S0_s])
    #print(outflow_u)
    #print(len(outflow_u))
    # add lower storage
    S0_l = par[pnames.index('S0_l')]; k2 = par[pnames.index('k2')]
    outflow_l, states_l = lb.unit_hbv_lower_storage(outflow_s,[S0_l,k2]) 
    #
    tp=par[pnames.index('tp')];k=par[pnames.index('k')]
    # Modelling transport
    streamflow=lb.unit_hydrograph(outflow_u+outflow_s+outflow_l+outflow_Y1C,[tp, k])
    #
    baseflow=par[pnames.index('baseflow')]
    #
    streamflow=streamflow+baseflow
    return(streamflow)


##### SSE ###
def sse(par_scale,pscale,pnames,data_input):
    #convert the scaled coefficients back to their original values
    par_unscale=[x*y for x,y in zip(par_scale,pscale)]
    #call the model function to generate a prediction for the given set of
    #parameters
    pred=simple_model(par_unscale, pnames, data_input)
    #extract the flow observations and convert them from pandas series
    #to numpy vector (predictions are also generated as numpy vector)
    flobs= data_input['flow'].to_numpy()
    flobs=flobs
    sse=np.nansum(np.power(np.subtract(flobs,pred),2))
    print(sse)
    
    
    return sse