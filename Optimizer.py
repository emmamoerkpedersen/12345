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
pmin={'Smaxsoil':0,'msoil':0.3,
      'betasoil':2,'cf':0.1,
      'baseflow':0,'S0soil':0,
      'tp':24,'k':1,'Smax_Y1C':1, 
      'PERC': 0, 'k0': 0.5 , 
      'k1':1, 'S0_s':0.1 ,
      'Smax_s':0, 'S0_l':11, 'k2':10}

pmax={'Smaxsoil':200,'msoil':1,
      'betasoil':20,'cf':1,
      'baseflow':100,'S0soil':10,
        'tp':120,'k':100,
        'Smax_Y1C':100, 'PERC': 100, 
        'k0':20, 'k1':100, 
        'S0_s':10,'Smax_s':100,
        'S0_l':1000, 'k2':100}

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
estimated_par = {pnames: par_estimate_unscaled for pnames, par_estimate_unscaled in estimated_par_zip}



# Calculate residuals
residuals = train['Precipitation']-train['Predict']

###plot
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(train['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(train['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(train['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')


##s########## Using the model on test data
pred_test=simple_model(par_estimate_unscaled, pnames,test)
#Add the predicted values to the train data
test['Predict'] = pred_test

# Calculate residuals
residuals = test['Precipitation']-test['Predict']

###plot
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(test['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(test['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(test['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011'))
ax[2].plot(residuals);ax[2].set_ylabel('Residuals')
