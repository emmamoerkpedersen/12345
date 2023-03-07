from SimpleModel import simple_model, sse, train_validate_test_split, sse_trace_val
import numpy as np
import matplotlib.pyplot as plt
import ddsoptim
import pandas as pd


####################################################
# Load data
data_in=pd.read_pickle('dataframe2.pkl')


####################################################
# DDS
## Set starting values, their ranges and scale
p0={'Smaxsoil':10,'msoil':1,
    'betasoil':19,'cf':0.5,
    'baseflow':3,'S0soil':10,
    'tp':2,'k':5,'Smax_Y1C':10, 
    'PERC': 50, 'k0':15, 'k1':50, 
    'S0_s':1 ,'Smax_s':50, 
    'S0_l':100, 'k2':99}

#convert dictionary to lists that are used as input to the model function
pmin={'Smaxsoil':10,'msoil':0.3,
      'betasoil':1,'cf':0.1,
      'baseflow':0,'S0soil':0,
      'tp':0,'k':1,'Smax_Y1C':1, 
      'PERC': 0, 'k0': 0.5 , 
      'k1':1, 'S0_s':0.1 ,
      'Smax_s':0, 'S0_l':11, 'k2':5}

pmax={'Smaxsoil':2000,'msoil':10,
      'betasoil':20,'cf':10,
      'baseflow':100,'S0soil':1000,
        'tp':48,'k':10,
        'Smax_Y1C':100, 'PERC': 100, 
        'k0':20, 'k1':100, 
        'S0_s':10,'Smax_s':100,
        'S0_l':1000, 'k2':100}

#  Parameters for optimizing
pscale={'Smaxsoil':1,'msoil':1,
        'betasoil':1,'cf':1,
        'baseflow':1,'S0soil':1,
        'tp':1,'k':1,'Smax_Y1C':1, 
        'PERC': 1, 'k0':1, 'k1':1, 
        'S0_s':1 ,'Smax_s':1, 
        'S0_l':1, 'k2':1}


pnames=list(p0.keys())
p0=list(p0.values())
pscale=list(pscale.values())

pmin=np.array(list(pmin.values()))
pmax=np.array(list(pmax.values()))

# Split data into train, validate and test. First half year is warm up hence the index 182
train, validate, test = train_validate_test_split(data_in[182:])
# Get the indices where train and validate start. The numbers are used in SSE
train_index = [data_in.index.get_loc(str(train.index[0].date())), data_in.index.get_loc(str(train.index[-1].date()))]
validate_index = [data_in.index.get_loc(str(validate.index[0].date())), data_in.index.get_loc(str(validate.index[-1].date()))]

#call ddsoptim - see ddsoptim.py for an explanation of the input arguments
par_estimate_unscaled,ssetrace=ddsoptim.ddsoptim(sse,p0,pmax,pmin,6711,0.2,True,pscale,pnames,data_in)
#best parameter value
np.nanmin(ssetrace)
np.nanmin(sse_trace_val)
#with dds the objective function is very noisy due to random sampling of parameters. we can compute a rolling min to
#see how the model fit improves with increasing number of iterations
plt.plot((pd.Series(ssetrace).rolling(50).min().to_numpy()), label = 'Train data')
plt.plot((pd.Series(sse_trace_val).rolling(500).min().to_numpy()), label = 'Validation')
plt.xlabel('No. of iterations')
plt.ylabel('MSE Error')
plt.legend()
plt.show()
### To get a list of the estimated parameters and their value
# Zip the names and values together
estimated_par_zip = zip(pnames, par_estimate_unscaled)
# Create a dictionary comprehension with the name-value tuples
par_estimate_list = {pnames: par_estimate_unscaled for pnames, par_estimate_unscaled in estimated_par_zip}

####################################################
############ Run the model for train and validation
#generate prediction from the model using the final parameter estimate
wTrain = data_in.loc['2011-09-29':'2019-01-21']

pred=simple_model(par_estimate_unscaled, pnames, wTrain) # INCLUDE WARMUP
#Add the predicted values to the train data
wTrain['Predict'] = pred

#generate prediction from the model using the final parameter estimate
pred_validate=simple_model(par_estimate_unscaled, pnames,validate)
#Add the predicted values to the train data
validate['Predict'] = pred_validate


####################################################
############ Aggregated data both for train and validation

############ Train
#aggregate to monthly resolution
#create an index that for each day indicates whether it belongs to the first month, the second month, and so on
factor=30
aggr_index=pd.Series(range(int((wTrain.shape[0]+1)/factor)))
aggr_index=aggr_index.repeat(factor)
aggr_index=aggr_index.reset_index(drop=True)
aggr_index.head()

#use this index in a grouping operation. average all values with the same index
wTrain=wTrain.iloc[0:aggr_index.shape[0],:]
wTrain['aggr_ind']=aggr_index.values
wTrain_agg=wTrain.groupby(['aggr_ind']).mean()

#get a time index for the aggregated values - we use the last day for each 30 days that are being aggregated
timewTrain=pd.DataFrame(wTrain.index)
timewTrain['aggr_ind']=aggr_index.values
timewTrain_agg=timewTrain.groupby(['aggr_ind']).tail(1)

#assign this new time index to the aggregated dataframe
wTrain_agg=wTrain_agg.set_index(timewTrain_agg['date'])
# Gathered in monthly resolution
residuals_agg = wTrain_agg['Precipitation']-wTrain_agg['Predict']


############ Validate
#aggregate to monthly resolution 
#create an index that for each day indicates whether it belongs to the first month, the second month, and so on
factor=30
aggr_index=pd.Series(range(int((validate.shape[0]+1)/factor)))
aggr_index=aggr_index.repeat(factor)
aggr_index=aggr_index.reset_index(drop=True)
aggr_index.head()

#use this index in a grouping operation. average all values with the same index
validate=validate.iloc[0:aggr_index.shape[0],:]
validate['aggr_ind']=aggr_index.values
validate_agg=validate.groupby(['aggr_ind']).mean()

#get a time index for the aggregated values - we use the last day for each 30 days that are being aggregated
timevalidate=pd.DataFrame(validate.index)
timevalidate['aggr_ind']=aggr_index.values
timevalidate_agg=timevalidate.groupby(['aggr_ind']).tail(1)

#assign this new time index to the aggregated dataframe
validate_agg=validate_agg.set_index(timevalidate_agg['date'])
# Gathered in monthly resolution
residuals_agg_val = validate_agg['Precipitation']-validate_agg['Predict']


############ Plotting all data and prediction
pred_all=simple_model(par_estimate_unscaled, pnames,data_in)
data_in['Predict'] = pred_all
residuals_all = data_in['flow']-data_in['Predict']

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(data_in['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(data_in['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(data_in['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011-08'))
ax[2].plot(residuals_all);ax[2].set_ylabel('Residuals')
plt.suptitle('Main')
ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # train period
ax[1].axvline(x=pd.to_datetime('2019-02-15'), color='k', linestyle='--') # validate period
ax[1].axvline(x=pd.to_datetime('2021-02-03'), color='k', linestyle='--') # Test period
plt.show()


## wTrain data
# Calculate residuals
residuals = wTrain['flow']-wTrain['Predict']

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(wTrain['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(wTrain['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(wTrain['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011-08'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
plt.suptitle('wTrain main plot')
ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # Warmup period
plt.show()

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(wTrain['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(wTrain['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(wTrain['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2016'), right=pd.to_datetime('2017'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
plt.suptitle('wTrain  zoomed')
plt.show()

# Plot aggregated values
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(wTrain_agg['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(wTrain_agg['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(wTrain_agg['Predict'])
ax[2].plot(residuals_agg);ax[2].set_ylabel('Residuals')
plt.suptitle('wTrain aggregated values main plot')
plt.show()

# Scatter obs. vs. residuals not aggregated
plt.plot(wTrain['flow'], 1 * wTrain['flow'])
plt.scatter(wTrain['flow'], wTrain['Predict'])
plt.xlabel('Observation [mm/day]'); plt.ylabel('Predicted [mm/day]')
plt.suptitle('Observation vs. Predicted - Train not aggregated')
plt.show()

# Scatter obs. vs. residuals aggregated
plt.scatter(wTrain_agg['Precipitation'], residuals_agg)
plt.xlabel('Precipitation'); plt.ylabel('Residuals')
plt.axhline(y=0)
plt.suptitle('Precipitation vs. residuals - wTrain aggregated')
plt.show()
# Scatter obs. vs. residuals not aggregated
residuals = wTrain['Precipitation']-wTrain['Predict']
plt.scatter(wTrain['Precipitation'], residuals)
plt.axhline(y=0)
plt.xlabel('Observed values'); plt.ylabel('Predicted values')
plt.suptitle('Precipitation vs. residuals - Train not aggregated')
plt.show()


####################################################
########## Validation plots
# Calculate residuals
residuals_validate = validate['flow']-validate['Predict']

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(validate['flow'], label= 'Obs');ax[1].set_ylabel('Flow')
ax[1].plot(validate['Predict'], label = 'Pred')
ax[1].set_xlim(left=pd.to_datetime('2019'))
ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
plt.suptitle('Validate main plot')
plt.show()

#### Zoom to peak periods
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(validate['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(validate['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2019-05'), right = pd.to_datetime('2019-11'))
ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
plt.suptitle('Validate - Peak 1')
plt.show()

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(validate['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(validate['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2020-05'), right = pd.to_datetime('2020-10'))
ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
plt.suptitle('Validate - Peak 2')
plt.show()

# Plot aggregated values
fig, ax = plt.subplots(3, 1, sharex=True,)
ax[0].plot(validate_agg['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(validate_agg['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(validate_agg['Predict'])
ax[2].plot(residuals_agg_val);ax[2].set_ylabel('Residuals')
plt.suptitle('Aggregated validate main plot')
plt.show()

###### Scatter aggregated values
# Scatter obs. vs. residuals not aggregated

plt.plot(validate['flow'], 1 * validate['flow'])
plt.scatter(validate['flow'], validate['Predict'])
plt.xlabel('Observation [mm/day]'); plt.ylabel('Predicted [mm/day]')
plt.suptitle('Observation vs. Predicted - Validate not aggregated')
plt.show()

# Scatter obs. vs. residuals aggregated
plt.scatter(validate_agg['Precipitation'], residuals_agg_val)
plt.xlabel('Precipitation'); plt.ylabel('Residuals')
plt.axhline(y=0)
plt.suptitle('Precipitation vs. residuals - Validate aggregated')
plt.show()
# Scatter obs. vs. residuals not aggregated
# Calculate residuals
residuals_validate = validate['flow']-validate['Predict']
plt.axhline(y=0)
plt.scatter(validate['Precipitation'], residuals_validate)
plt.xlabel('Precipitation'); plt.ylabel('Residuals')
plt.suptitle('Precipitation vs. residuals - Validate not aggregated')
plt.show()


####################################################
## Test
pred_test=simple_model(par_estimate_unscaled, pnames,test)
test['Predict'] = pred_test
residuals_test = test['flow']-test['Predict']

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(test['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(test['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(test['Predict'])
ax[2].plot(residuals_test);ax[2].set_ylabel('Residuals')
plt.suptitle('test main plot')
plt.show()
####################################################
## Residuals

# Histogram
plt.hist(residuals, bins = 15)
plt.xlim(left = -30, right = 30)
plt.show()

# Autocorrelation
from statsmodels.graphics import tsaplots
tsaplots.plot_acf(residuals, lags=20)
plt.show()


plt.hist(residuals_validate, bins = 15)
plt.xlim(left = -30, right = 30)
plt.show()

tsaplots.plot_acf(residuals_validate, lags=20)
plt.show()

plt.hist(residuals_test, bins = 15)
plt.xlim(left = -10, right = 10)
plt.show()

tsaplots.plot_acf(residuals_test, lags=20)
plt.show()