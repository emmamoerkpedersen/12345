import numpy as np

states = {}
outflow_dict = ()
####################################
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
    # #####################
    #add shallow storage
    Smax_s=par[pnames.index('Smax_s')]
    outflow_s, percolation_s, states_s = lb.unit_hbv_shallow_storage(outflow_u,[Smax_s,PERC,k0,k1,S0_s])

    #add lower storage
    # S0_l = par[pnames.index('S0_l')]; k2 = par[pnames.index('k2')]
    # outflow_l, states_l = lb.unit_hbv_lower_storage(outflow_s,[S0_l,k2]) 
    
    tp=par[pnames.index('tp')];k=par[pnames.index('k')]
    # Modelling transport
    streamflow=lb.unit_hydrograph(outflow_u+outflow_s  + outflow_Y1C,[tp, k])
    #
    baseflow=par[pnames.index('baseflow')]
    #
    streamflow=streamflow+baseflow
    
    states['states_u'] = states_u
    #states['states_Y1C'] = states_Y1C
    states['states_s'] = states_s
    #states['states_l'] = states_l
    
    return(streamflow)

sse_trace_val = []
##### SSE ###
def sse(par_scale,pscale,pnames, data_input):
    import pandas as pd
    #convert the scaled coefficients back to their original values
    par_unscale=[x*y for x,y in zip(par_scale,pscale)]
    #call the model function to generate a prediction for the given set of
    #parameters
    pred=simple_model(par_unscale, pnames, data_input)
    #extract the flow observations and convert them from pandas series
    #to numpy vector (predictions are also generated as numpy vector)
    flobs= data_input['flow'].to_numpy()

    train_index = [182, 2697]
    validate_index = [2698, 3388]
    # Calculate sse for training
    sse=np.nansum(np.power(np.subtract(flobs[train_index[0]:train_index[1]],pred[train_index[0]:train_index[1]]),2))/2515
    
    global sse_trace_val
    sse_val=(np.nansum(np.power(np.subtract(flobs[validate_index[0]:validate_index[1]],pred[validate_index[0]:validate_index[1]]),2)))/690
    sse_trace_val.append(sse_val)

    #print(sse)
    #print(len(sse_trace_val))

    return sse


###### splitting in train, val and test data
def train_validate_test_split(df, train_percent=.7, validate_percent=.2, seed=None):
    #np.random.seed(seed)
    perm = df.index
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    train = train.sort_values('date')
    validate = df.loc[perm[train_end:validate_end]]
    validate = validate.sort_values('date')
    test = df.loc[perm[validate_end:]]
    test = test.sort_values('date')
    return train, validate, test


#### Code only to make state plot
# par_estimate_unscaled,ssetrace=ddsoptim.ddsoptim(sse,p0,pmax,pmin,2500,0.2,True,pscale,pnames,data_in)
# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(len(states), 1, figsize=(10, 10))
# for i, (key, value) in enumerate(states.items()):
#     axs[i].plot(value['S'])
#     axs[i].set_title(key)
# plt.tight_layout()
# plt.show()