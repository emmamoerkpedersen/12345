from SimpleModel import simple_model, p0
from LoadData import train, test, validate, data_Average
import numpy as np

########################

def sse(par_scale,pscale,pnames,train):
    #convert the scaled coefficients back to their original values
    par_unscale=[x*y for x,y in zip(par_scale,pscale)]
    #call the model function to generate a prediction for the given set of
    #parameters
    pred=simple_model(par_unscale, pnames, train)
    #extract the flow observations and convert them from pandas series
    #to numpy vector (predictions are also generated as numpy vector)
    flobs= train['flow'].to_numpy()
    flobs=flobs
    sse=np.nansum(np.power(np.subtract(flobs,pred),2))
    print(sse)
    
    
    return sse
####################################################
#train = train.reset_index(drop = True)[['Precipitation', 'flow']]

#estimate parameters using DDS
import ddsoptim


p0 = {'Smaxsoil':1,'msoil':1,'betasoil':2,'cf':0.1,'baseflow':3,'S0soil':1,'tp':2,'k':10,'Smax_Y1C':1000, 'PERC': 100, 'k0':250, 'k1':100, 'S0_s':0.1 ,'Smax_s':1000, 'S0_l':1000, 'k2':1}
# Parameters for optimizing
pscale = {'Smaxsoil':1,'msoil':1,'betasoil':1,'cf':1,'baseflow':1,'S0soil':1,'tp':1,'k':1,'Smax_Y1C':1, 'PERC': 1, 'k0':1, 'k1':1, 'S0_s':1 ,'Smax_s':1, 'S0_l':1, 'k2':1}
#convert dictionary to lists that are used as input to the model function
pmin={'Smaxsoil':0,'msoil':0,'betasoil':2,'cf':0.1,'baseflow':3,'S0soil':1,'tp':1,'k':1,'Smax_Y1C':1, 'PERC': 1, 'k0': 1 , 'k1':1, 'S0_s':0.1 ,'Smax_s':1, 'S0_l':11, 'k2':1}
pmax={'Smaxsoil':100,'msoil':100,'betasoil':20,'cf':1,'baseflow':30,'S0soil':10, 'tp':20,'k':100,'Smax_Y1C':100, 'PERC': 100, 'k0':250, 'k1':100, 'S0_s':10,'Smax_s':1 ,'S0_l':1000, 'k2':10}

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


#plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(train['Precipitation']);ax[0].set_ylabel('Rain')
ax[1].plot(train['flow']);ax[1].set_ylabel('Flow')
ax[1].plot(train['Predict'])
ax[1].set_xlim(left=pd.to_datetime('2011'))




















theta0=p0
#parameter scale (guess what order of magnitude each parameter will have)
scale=np.array([1,10])
#
from scipy.optimize import minimize
p0_scale = [x*y for x,y in zip(p0,pscale)]
res = minimize(fun=sse, x0=p0_scale, args=(pscale,pnames,train), method='Nelder-Mead', jac=False,options={'disp': True,'maxiter':100})



#res = minimize(fun=sse, x0=theta0, args=(scale,data), method='Nelder-Mead', jac=False,options={'disp': True,'maxiter':10000})
par_estimate_unscaled=[x*y for x,y in zip(res.x,pscale)]


#generate prediction from the model using the final parameter estimate
pred=simple_model(par_estimate_unscaled,pnames,train)
plt.plot(pred)

