import numpy as np
import matplotlib.pyplot as plt
import ddsoptim
import pandas as pd
from SimpleModel import simple_model, sse, train_validate_test_split, sse_trace_val

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


####################################################

#call ddsoptim - see ddsoptim.py for an explanation of the input arguments
par_estimate_unscaled,ssetrace=ddsoptim.ddsoptim(sse,p0,pmax,pmin,2500,0.2,True,pscale,pnames,data_in)
#best parameter value
np.nanmin(ssetrace)
np.nanmin(sse_trace_val)
plt.plot(ssetrace)
plt.show()
#with dds the objective function is very noisy due to random sampling of parameters. we can compute a rolling min to
#see how the model fit improves with increasing number of iterations
plt.plot((pd.Series(ssetrace).rolling(500).mean().to_numpy()), label = 'Train data')
plt.plot((pd.Series(sse_trace_val).rolling(500).mean().to_numpy()), label = 'Validation')
plt.xlim(500, 2000)
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


############ Plotting
liste = [data_in, test]
liste_names = ['All data', 'Test']

for i in range(len(liste)):
      # Run the model and predict values
      pred=simple_model(par_estimate_unscaled, pnames, liste[i])
      # Save the prediction to the list
      liste[i]['Predict'] = pred
      # Calculate residuals
      residuals = liste[i]['flow']-liste[i]['Predict']

      ## Plot containing precipitation, obs vs. pred and residuals
      fig, ax = plt.subplots(3, 1, sharex=True, figsize = (15,8))
      ax[0].plot(liste[i]['Precipitation']);ax[0].set_ylabel('Rain')
      ax[1].plot(liste[i]['flow']);ax[1].set_ylabel('Flow')
      ax[1].plot(liste[i]['Predict'])
      #ax[1].set_xlim(left=pd.to_datetime('2011-08'))
      ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
      plt.suptitle(liste_names[i])
      if liste_names[i] == 'All data':
             # Shade areas for different data periods
             ax[1].axvspan(pd.to_datetime('2011-10-01'), pd.to_datetime('2012-03-29'), facecolor='red', alpha=0.2)  # Warmup period
             ax[1].axvspan(pd.to_datetime('2012-03-29'), pd.to_datetime('2019-02-15'), facecolor='blue', alpha=0.2)  # Train period
             ax[1].axvspan(pd.to_datetime('2019-02-15'), pd.to_datetime('2021-02-03'), facecolor='green', alpha=0.2)  # Validate period
             ax[1].axvspan(pd.to_datetime('2021-02-03'), pd.to_datetime('2022-01-01'), facecolor='orange', alpha=0.2)  # Test period

        # Create legend patches
             #warmup_patch = mpatches.Patch(color='red', alpha=0.2, label='Warmup')
             #train_patch = mpatches.Patch(color='blue', alpha=0.2, label='Train')
             #validate_patch = mpatches.Patch(color='green', alpha=0.2, label='Validate')
             #test_patch = mpatches.Patch(color='orange', alpha=0.2, label='Test')

        # Add legend to the plot
             #ax[1].legend(handles=[warmup_patch, train_patch, validate_patch, test_patch],loc='upper left')
             ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # train period
             ax[1].axvline(x=pd.to_datetime('2019-02-15'), color='k', linestyle='--') # validate period
             ax[1].axvline(x=pd.to_datetime('2021-02-03'), color='k', linestyle='--') # Test period
             plt.show()
      
      elif liste_names[i] == 'Warmup and Train':
            ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # Warmup period
            plt.show()
      else:
            plt.show()

      # Scatterplots Observation vs. predict
      plt.figure(figsize=(10,8))
      plt.plot(liste[i]['flow'], 1*liste[i]['flow'])
      plt.scatter(liste[i]['flow'], liste[i]['Predict'])
      plt.xlabel('Observation [mm/day]'); plt.ylabel('Predicted [mm/day]')
      plt.suptitle(f'Observation vs. Predicted - {liste_names[i]}')
      plt.show()

      ## Scatterplots precipitations vs. residuals
      plt.figure(figsize=(10,8))
      plt.scatter(liste[i]['Precipitation'], residuals)
      plt.xlabel('Precipitation'); plt.ylabel('Residuals')
      plt.axhline(y=0)
      plt.suptitle(f'Precipitation vs. residuals - {liste_names[i]}')
      plt.show()
      
      ## Plot histogram for residuals
      plt.figure(figsize=(10,8))
      residuals = liste[i]['flow']-liste[i]['Predict']
      plt.hist(residuals, bins = 15)
      plt.suptitle(f'Histogram for {liste_names[i]} residuals')
      plt.show()

      ## Plot Autocorrelation for residuals
      from statsmodels.graphics import tsaplots
      tsaplots.plot_acf(residuals, lags=20)
      plt.suptitle(f'Autocorrelation for {liste_names[i]} residuals')
      plt.show()

####################################################
############ Aggregated the data and split again
# factor=30
# aggr_index=pd.Series(range(int((data_in.shape[0]+1)/factor)))
# aggr_index=aggr_index.repeat(factor)
# aggr_index=aggr_index.reset_index(drop=True)
# aggr_index.head()

# #use this index in a grouping operation. average all values with the same index
# data_in=data_in.iloc[0:aggr_index.shape[0],:]
# data_in['aggr_ind']=aggr_index.values
# data_in_agg=data_in.groupby(['aggr_ind']).mean()

# #get a time index for the aggregated values - we use the last day for each 30 days that are being aggregated
# timeData=pd.DataFrame(data_in.index)
# timeData['aggr_ind']=aggr_index.values
# timeData_agg=timeData.groupby(['aggr_ind']).tail(1)

# #assign this new time index to the aggregated dataframe
# data_in_agg=data_in_agg.set_index(timeData_agg['date'])
# Gathered in monthly resolution

# Split the aggregated data into train+warmup, validate and test
# wTrain_agg, validate_agg, test_agg = train_validate_test_split(data_in_agg[:182])

# # Calculate residuals for the aggregated data
# residuals_agg = wTrain_agg['Precipitation']-wTrain_agg['Predict']
# residuals_agg_val = validate_agg['Precipitation']-validate_agg['Predict']
# residuals_agg_test = test_agg['Precipitation']-test_agg['Predict']
#residuals_agg_all = data_in_agg['Precipitation']-data_in_agg['Predict']


#########################################
## Plot aggregated values
# liste_agg = [data_in_agg, wTrain_agg , validate_agg, test_agg]
# liste_names = ['All data monthly', 'Warmup and Train monthly' , 'Validate monthly', 'Test monthly']

# for i in range(len(liste)):
#       pred=simple_model(par_estimate_unscaled, pnames, liste_agg[i])
#       liste_agg[i]['Predict'] = pred
#       residuals = liste_agg[i]['flow']-liste_agg[i]['Predict']

#       # Plots of aggregated data
#       fig, ax = plt.subplots(3, 1, sharex=True)
#       ax[0].plot(liste_agg[i]['Precipitation']);ax[0].set_ylabel('Rain')
#       ax[1].plot(liste_agg[i]['flow']);ax[1].set_ylabel('Flow')
#       ax[1].plot(liste_agg[i]['Predict'])
#       #ax[1].set_xlim(left=pd.to_datetime('2011-08'))
#       ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
#       plt.suptitle(liste_names[i])
#       if liste_names[i] == 'All data monthly':
#             ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # train period
#             ax[1].axvline(x=pd.to_datetime('2019-02-15'), color='k', linestyle='--') # validate period
#             ax[1].axvline(x=pd.to_datetime('2021-02-03'), color='k', linestyle='--') # Test period
#             plt.show()
#       elif liste_names[i] == 'Warmup and Train montly':
#             ax[1].axvline(x=pd.to_datetime('2012-03-29'), color='k', linestyle='--') # Warmup period
#             plt.show()
#       else:
#             plt.show()
      
#       # Scatterplots of aggregated data
#       plt.scatter(liste_agg[i]['Precipitation'], residuals)
#       plt.xlabel('Precipitation'); plt.ylabel('Residuals')
#       plt.axhline(y=0)
#       plt.suptitle(f'Precipitation vs. residuals - {liste_names[i]} monthly')
#       plt.show()

#########################################
# Calculate residuals
residuals = wTrain['flow']-wTrain['Predict']
residuals_validate = validate['flow']-validate['Predict']
residuals_test = test['flow']-test['Predict']

# ## Plot zoomed 
# ## wTrain data
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(wTrain['Precipitation']);ax[0].set_ylabel('Rain')
# ax[1].plot(wTrain['flow']);ax[1].set_ylabel('Flow')
# ax[1].plot(wTrain['Predict'])
# ax[1].set_xlim(left=pd.to_datetime('2016'), right=pd.to_datetime('2017'))
# ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
# plt.suptitle('wTrain  zoomed')
# plt.show()

# #### Validate
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
# ax[1].plot(validate['flow']);ax[1].set_ylabel('Flow')
# ax[1].plot(validate['Predict'])
# ax[1].set_xlim(left=pd.to_datetime('2019-05'), right = pd.to_datetime('2019-11'))
# ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
# plt.suptitle('Validate - Peak 1')
# plt.show()

# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
# ax[1].plot(validate['flow']);ax[1].set_ylabel('Flow')
# ax[1].plot(validate['Predict'])
# ax[1].set_xlim(left=pd.to_datetime('2020-05'), right = pd.to_datetime('2020-10'))
# ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
# plt.suptitle('Validate - Peak 2')
# plt.show()


# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(validate['Precipitation']);ax[0].set_ylabel('Rain')
# ax[1].plot(validate['flow']);ax[1].set_ylabel('Flow')
# ax[1].plot(validate['Predict'])
# ax[1].set_xlim(left=pd.to_datetime('2020-05'), right = pd.to_datetime('2020-10'))
# ax[2].plot(residuals_validate);ax[2].set_ylabel('Residuals')
# plt.suptitle('Validate - Peak 2')
# plt.show()

# Calcualte NSE
obs = test['flow']
pred = test['Predict']
mean_obs = test['flow'].mean()

numerator = np.sum((obs - pred) ** 2)
denominator = np.sum((obs - mean_obs) ** 2)

nse = 1 - (numerator / denominator)





