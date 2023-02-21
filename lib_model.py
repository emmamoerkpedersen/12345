import numpy as np
from numba import njit

@njit
def lower_b(val,lb=0.0):
    #clip a value or array at 0 
    out=((val-lb)/(1.0+np.exp(1e3*(-(val-lb)))*1.0))+lb+0.0003
    return(out)

@njit
def compute_snow_storage_eq(inflow,S0,potential_melt,lower_limit,m):
    #compute storage changes, imposing upper and lower limits
    storage=np.zeros(inflow.shape[0])
    storage[0]=max(0,S0)
    outflow=np.zeros(inflow.shape[0])
    for ii in range(1,inflow.shape[0]):
        melt=max(0,potential_melt[ii]*(1-np.exp(-storage[ii-1]/m)))
        sdiff=storage[ii-1]+inflow[ii]-melt
        if sdiff<lower_limit:
            storage[ii]=lower_limit
            outflow[ii]=storage[ii-1]+inflow[ii]
        else:
            storage[ii]=sdiff
            outflow[ii]=melt
    return(storage,outflow)

def unit_snow_storage(T,P,par):
    #models snow storage in a catchment
    #see https://superflexpy.readthedocs.io/en/latest/elements_list.html
    #outputs rain+melt if T>melting temperature, accumulates snow in storage otherwise
    #input series: T=temperature in degC, P=rainfall in mm/day
    #parameters: par[0]=melt temperature in degC
    #par[1]=k, such that k*T = potential melt rate in mm/day
    #par[2] exponent that controls how much the melt rate is reduced when storage is low
    Tsnow=np.float(par[0])
    ksnow=lower_b(np.float(par[1]))
    betasnow=lower_b(np.float(par[2]))
    S0=lower_b(np.float(par[3])) #initial storage
    #
    melt=np.zeros(P.shape[0])
    melt[T>=Tsnow]=ksnow*T[T>=Tsnow]-Tsnow #melt rate depends on how many degrees we are above the melting point
    snowacc=np.zeros(P.shape[0])
    snowacc[T<Tsnow]=P[T<Tsnow]
    snowstorage,snowoutflow=compute_snow_storage_eq(snowacc,S0,melt,0,betasnow)
    outflow=snowoutflow
    outflow[T>=Tsnow]=outflow[T>=Tsnow]+P[T>=Tsnow]
    states={'S':snowstorage}
    #
    return(outflow,states)
    

@njit
def compute_soil_zone_storage(I_s,PET_s,S0,Smax,m,beta,return_ET=False):
    storage=np.zeros(I_s.shape[0])
    storage[0]=max(0,S0)
    outflow=np.zeros(I_s.shape[0])
    if return_ET: evapotranspiration=np.zeros(I_s.shape[0])
    for ii in range(1,I_s.shape[0]):
        Sratio=min(1.0,storage[ii-1]/Smax)
        EACT=PET_s[ii]*(Sratio*(1+m)/(Sratio+m))
        Q=I_s[ii]*(1-(1-Sratio)**beta)
        storage[ii]=max(0,storage[ii-1]+I_s[ii]-EACT-Q)
        outflow[ii]=Q
        if return_ET: evapotranspiration[ii]=EACT
    #
    if return_ET:
        return(outflow,[storage,evapotranspiration])
    else:
        return(outflow,[storage])
    

def unit_soil_zone_storage(PET,I,par,return_ET=False):
    #models soil zone storage in a catchment
    #see https://superflexpy.readthedocs.io/en/latest/elements_list.html#upper-zone-hymod
    #outputs inflow-ET if storage is full, otherwise accumulates part of the inflow in storage
    #input series: PET=potential evapotranspiration in mm/day, P=inflow (effective rainfall, possibly outflow from snow storage) in mm/day
    #parameters: par[0]=Smax=storage capacity in mm
    #par[1]=m=adjustment coefficient for computing ET from PET and S (storage) - controls how fast ET approaches PET as the soil storage becomes full
    #par[2]=beta=exponent for computing outflow from storage
    Smax=lower_b(np.float(par[0]),1e-2)
    m=lower_b(np.float(par[1]))
    beta=lower_b(np.float(par[2]))
    S0=lower_b(np.float(par[3])) #initial storage
    #
    substeps=4
    PET_s=np.repeat(PET,substeps)/float(substeps)
    I_s=np.repeat(I,substeps)/float(substeps)
    #
    outflow, states=compute_soil_zone_storage(I_s,PET_s,S0,Smax,m,beta,return_ET=return_ET)
    #return last value from each substep interval
    index=np.arange(outflow.shape[0],step=substeps)
    if return_ET:
        statesd={'S':states[0][index],'ET':states[1][index]}
    else:
        statesd={'S':states[0][index]}
    #
    return(outflow[index]*substeps,statesd)

@njit
def compute_shallow_storage(I_s,S0,Smax,PERC,k0,k1):
    storage=np.zeros(I_s.shape[0])
    storage[0]=max(0,S0)
    streamflow=np.zeros(I_s.shape[0])
    percolation=np.zeros(I_s.shape[0])
    for ii in range(1,I_s.shape[0]):
        storage_step=storage[ii-1]+I_s[ii]
        Sratio=min(1,storage_step/Smax)
        percolation_step=min(PERC,storage_step)
        storage_step=max(0,storage_step-percolation_step)
        streamflow_step=storage_step/k1
        storage_step=max(0,storage_step-streamflow_step)
        spillflow_step=max(0,(storage_step-Smax)/k0)
        storage_step=max(0,storage_step-spillflow_step)
        storage[ii]=storage_step
        streamflow[ii]=streamflow_step+spillflow_step
        percolation[ii]=percolation_step
    #
    return([streamflow,percolation],[storage])


def unit_hbv_shallow_storage(I,par):
    #implements shallow subsurface storage element as described in 
    #Herman, J.D., Reed, P.M., Wagener, T., 2013. Time-varying sensitivity analysis clarifies the effects of watershed model formulation on model behavior. Water Resour. Res. 49, 1400–1414. https://doi.org/10.1002/wrcr.20124
    #takes inflow from e.g. upper zone and converts it to PERCOLATION (PERC) towards lower zone, direct streamflow (S/K1), spillover ((S-Smax)/k0)
    #generates two outflows: streamflow (direct+spillover), percolation to lower storage
    #parameters: par[0]=Smax=storage capacity [mm], any water exceeded Smax will be directed into the spillover outflow
    #par[1]=PERC=percolation rate [mm/d], fixed outflow rate towards lower storage
    #par[2]=k0=delay constant for spillover flow [d]
    #par[3]=k1=delay constant for direct streamflow [d]
    substeps=4
    #
    Smax=lower_b(np.float(par[0]),1e-2)
    PERC=lower_b(np.float(par[1])/float(substeps))
    k0=lower_b(np.float(par[2])/float(substeps))
    k1=lower_b(np.float(par[3])/float(substeps))
    S0=lower_b(np.float(par[4])) #initial storage
    #
    I_s=np.repeat(I,substeps)/float(substeps)
    #
    outflows,states=compute_shallow_storage(I_s,S0,Smax,PERC,k0,k1)
    #return last value from each substep interval
    index=np.arange(outflows[0].shape[0],step=substeps)    
    statesd={'S':states[0][index]}
    #streamflow,percolation,storage state
    #outflows need to be multiplied again by no. of substeps to get daily rates (otherwise we loose water)
    return(outflows[0][index]*substeps,outflows[1][index]*substeps,statesd)

@njit
def compute_lower_storage(I_s,S0,k2):
    storage=np.zeros(I_s.shape[0])
    storage[0]=max(0,S0)
    outflow=np.zeros(I_s.shape[0])
    outflow[0]=S0/k2
    for ii in range(1,I_s.shape[0]):
        storage_step=storage[ii-1]+I_s[ii]
        outflow[ii]=storage_step/k2
        storage[ii]=storage_step-outflow[ii]
    #
    return(outflow,[storage])

def unit_hbv_lower_storage(I,par):
    #implements lower storage reservoir. this reservoir has no storage limation, only an outflow that depends on current storage
    #parameters: par[0]=k2=outflow delay constant [d]
    substeps=4
    k2=lower_b(np.float(par[0])/float(substeps))
    S0=lower_b(np.float(par[1]))
    #
    I_s=np.repeat(I,substeps)/float(substeps)
    #
    outflow,states=compute_lower_storage(I_s,S0,k2)
    #return last value from each substep interval
    index=np.arange(outflow.shape[0],step=substeps)    
    statesd={'S':states[0][index]}
    #streamflow,percolation,storage state
    return(outflow[index]*substeps,statesd)    

def unit_model(tp, k):
    from scipy.stats import gamma
    #****helper function
    #create unit hydrograph for 600 time steps for a combination of tp and k
    #tp = time to peak measured from the runoff (excess rainfall) pulse, this corresponds to lag time in NEH15 AND NOT time to peak (measured from when the rainfall starts)
    #k - flatness of the hydrograph
    #see Viet Ha (https://findit.dtu.dk/en/catalog/2607128905) p. 10 - tp=(n-1)*k
    #NEH gives lag time as time from excess rainfall until time to peak
    time = np.arange(600)
    unit_hydro = gamma.cdf(time, tp/k+1, 0.0, k) - gamma.cdf(time-1, tp/k+1, 0.0, k)   
    return(unit_hydro)

def unit_hydrograph(runoff_series, par):
    #direct function that simulates flow towards the catchment runoff
    #expects a single inflow series (typically corresponding to the surface runoff)
    #output is the flow at the catchment outlet
    #parameters: tp = time to peak in days, this can be fixed based on flow length of catchment
    #k=delay constant in 1/d
    tp=par[0]
    k=par[1]
    outflow = np.convolve(unit_model(tp, k), runoff_series)
    outflow=outflow[0:len(runoff_series)]
    return(outflow)

