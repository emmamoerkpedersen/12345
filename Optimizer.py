from SimpleModel import simple_model
from LoadData import train, test, validate

########################

def sse(par_scale,pscale,pnames,train):
    #convert the scaled coefficients back to their original values
    par_unscale=[x*y for x,y in zip(par_scale,pscale)]
    #call the model function to generate a prediction for the given set of
    #parameters
    pred=simple_model(par_unscale, pnames, train)
    #extract the flow observations and convert them from pandas series
    #to numpy vector (predictions are also generated as numpy vector)
    flobs= train['flowY1C'].to_numpy()
    flobs=flobs
    sse=np.nansum(np.power(np.subtract(flobs,pred),2))
    print(sse)
    
    
    return sse
####################################################

#estimate parameters using DDS
import ddsoptim

#start parameters
theta0=np.array([1,100])
#DDS does not require parameter scaling. We just define a constant vector of ones here, so that we can used the 
#same sse function for both minimize and ddsoptim
scale=np.array([1,10])
#DDS does require you to specify upper and lower parameter bounds. these ranges need to cover only the range of values within
#which you expect the parameter values to be. DDS randomly samples from these ranges. If you make it too big, you
#reduce your chance of finding good parameter combinations
pmax=np.array([10,500])
pmin=np.array([1e-3,1])
#call ddsoptim - see ddsoptim.py for an explanation of the input arguments
par_estimate_unscaled,ssetrace=ddsoptim.ddsoptim(sse,theta0,pmax,pmin,7500,0.2,True,scale,train)
#best parameter value
np.nanmin(ssetrace)
#plot how objective function changes during optimization
ssetrace[ssetrace>2]=2
plt.plot(ssetrace)
#with dds the objective function is very noisy due to random sampling of parameters. we can compute a rolling min to
#see how the model fit improves with increasing number of iterations
plt.plot(pd.Series(ssetrace).rolling(50).min().to_numpy())

#generate prediction from the model using the final parameter estimate
pred=simplemod(par_estimate_unscaled,data['Rain'])
#plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(data['Rain']);ax[0].set_ylabel('Rain')
ax[1].plot(data['Flow']);ax[1].set_ylabel('Flow')
ax[1].plot(pred)






















theta0=np.array([1,100])
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

