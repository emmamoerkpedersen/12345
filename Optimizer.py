from SimpleModel import simple_model, sse
from LoadData import train, test, validate, data_Average
import numpy as np
import matplotlib.pyplot as plt
import ddsoptim
import pandas as pd

####################################################

# DDS
## Set starting values, their ranges and scale
p0={'Smaxsoil':1,'msoil':1,
    'betasoil':2,'cf':0.1,
    'baseflow':3,'S0soil':1,
    'tp':2,'k':10,'Smax_Y1C':1000, 
    'PERC': 100, 'k0':250, 'k1':100, 
    'S0_s':0.1 ,'Smax_s':1000, 
    'S0_l':1000, 'k2':1}

#  Parameters for optimizing
pscale={'Smaxsoil':1,'msoil':1,
        'betasoil':1,'cf':1,
        'baseflow':1,'S0soil':1,
        'tp':1,'k':1,'Smax_Y1C':1, 
        'PERC': 1, 'k0':1, 'k1':1, 
        'S0_s':1 ,'Smax_s':1, 
        'S0_l':1, 'k2':1}
#convert dictionary to lists that are used as input to the model function
pmin={'Smaxsoil':10,'msoil':0.3,
      'betasoil':2,'cf':0.1,
      'baseflow':0,'S0soil':0,
      'tp':0,'k':1,'Smax_Y1C':1, 
      'PERC': 0, 'k0': 0.5 , 
      'k1':1, 'S0_s':0.1 ,
      'Smax_s':0, 'S0_l':11, 'k2':10}

pmax={'Smaxsoil':2000,'msoil':10,
      'betasoil':20,'cf':10,
      'baseflow':100,'S0soil':1000,
        'tp':48,'k':10,
        'Smax_Y1C':100, 'PERC': 100, 
        'k0':20, 'k1':100, 
        'S0_s':10,'Smax_s':100,
        'S0_l':1000, 'k2':2000}

pnames=list(p0.keys())
p0=list(p0.values())
pscale=list(pscale.values())

pmin=np.array(list(pmin.values()))
pmax=np.array(list(pmax.values()))


#call ddsoptim - see ddsoptim.py for an explanation of the input arguments
par_estimate_unscaled,ssetrace=ddsoptim.ddsoptim(sse,p0,pmax,pmin,7500,0.2,True,pscale,pnames,train)
#best parameter value
np.nanmin(ssetrace)
#plot how objective function changes during optimization
ssetrace[ssetrace>2]=2
plt.plot(ssetrace)
#with dds the objective function is very noisy due to random sampling of parameters. we can compute a rolling min to
#see how the model fit improves with increasing number of iterations
plt.plot(pd.Series(ssetrace).rolling(50).min().to_numpy())

#generate prediction from the model using the final parameter estimate
pred=simple_model(par_estimate_unscaled, pnames,train)
#Add the predicted values to the train data
train['Predict'] = pred

### To get a list of the estimated parameters and their value
# Zip the names and values together
estimated_par_zip = zip(pnames, par_estimate_unscaled)

# Create a dictionary comprehension with the name-value tuples
par_estimate_list = {pnames: par_estimate_unscaled for pnames, par_estimate_unscaled in estimated_par_zip}



# Calculate residuals
residuals = train['Precipitation']-train['Predict']

######## PLOTS #######
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(train['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(train['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(train['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
plt.show()

############ Using the model on TEST data
pred_test=simple_model(par_estimate_unscaled, pnames,test)
#Add the predicted values to the train data
test['Predict'] = pred_test

# Calculate residuals
residuals_test = test['Precipitation']-test['Predict']
 
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(test['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(test['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(test['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
plt.show()

############ Aggregate the data

#aggregate to monthly resolution TRAIN
#create an index that for each day indicates whether it belongs to the first month, the second month, and so on
factor=30
aggr_index=pd.Series(range(int((train.shape[0]+1)/factor)))
aggr_index=aggr_index.repeat(factor)
aggr_index=aggr_index.reset_index(drop=True)
aggr_index.head()

#use this index in a grouping operation. average all values with the same index
train=train.iloc[0:aggr_index.shape[0],:]
train['aggr_ind']=aggr_index.values
train_agg=train.groupby(['aggr_ind']).mean()

#get a time index for the aggregated values - we use the last day for each 30 days that are being aggregated
timetrain=pd.DataFrame(train.index)
timetrain['aggr_ind']=aggr_index.values
timetrain_agg=timetrain.groupby(['aggr_ind']).tail(1)

#assign this new time index to the aggregated dataframe
train_agg=train_agg.set_index(timetrain_agg['date'])
# Gathered in monthly resolution
residuals_agg = train_agg['Precipitation']-train_agg['Predict']

# Plot aggregated values
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(train_agg['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(train_agg['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(train_agg['Predict'])
ax[2].plot(residuals_agg);ax[2].set_ylabel('Residuals')
plt.show()

###### Scatter aggregated values
plt.scatter(train_agg['Precipitation'],train_agg['Predict'])
plt.xlabel('Observed values'); plt.ylabel('Predicted values')
plt.axline((0,0), slope=0.5, color='k', transform=plt.gca().transAxes)
plt.title('Aggregated')
plt.show()
###### Scatter
plt.scatter(train['Precipitation'], train['Predict'])
plt.xlabel('Observed values'); plt.ylabel('Predicted values')
plt.axline((0,0), slope=0.5, color='k', transform=plt.gca().transAxes)
plt.title('Not aggregated')
plt.show()

# Scatter obs. vs. residuals
residuals = train['Precipitation']-train['Predict']
plt.scatter(train['Precipitation'], residuals)
plt.xlabel('Observed values'); plt.ylabel('Predicted values')
plt.axline((0,0), slope=0.5, color='k', transform=plt.gca().transAxes)
plt.title('Not aggregated')
plt.show()

# Scatter obs. vs. residuals aggregated
plt.scatter(train_agg['Precipitation'], residuals_agg)
plt.xlabel('Observed values'); plt.ylabel('Predicted values')
plt.axline((0,0), slope=0.5, color='k', transform=plt.gca().transAxes)
plt.title('Aggregated')
plt.show()
