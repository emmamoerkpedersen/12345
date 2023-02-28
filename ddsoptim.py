def xreflect(x,xmin,xmax):
    select=x<xmin
    x[select]=xmin[select]+(xmin[select]-x[select])
    x[select][x[select]>xmax[select]]=xmin[select][x[select]>xmax[select]]
    select=x>xmax
    x[select]=xmax[select]-(x[select]-xmax[select])
    x[select][x[select]<xmin[select]]=xmax[select][x[select]<xmin[select]]
    return(x)

def ddsoptim(f,start,xmax,xmin,ndds=2500,rdds=0.2,decrvalrange=False,*args):
  #DDS search algorithm to find not the optimal but a good set of parameters
  #Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. Water Resources Research, 43(1), W01413. https://doi.org/10.1029/2005WR004723
  #f=function to optimize, the first input argument to this function must be the parameter vector
  #start=starting parameters
  #ndds=no. of objective function evaluations to perform
  #rdds=perturbation parameter - DO NOT TOUCH UNLESS YOU KNOW WHAT YOU'RE DOING
  #xmax, xmin=vectors of max and min values for each parameter (define reasonable ranges, the algorithm searches these intervals)
  #decrvalrange-if true, not only the number of parameters that are modified in each iteration is reduced as the optimization proceeds, but also the range within which each parameter is modified
  #*args - any other input arguments that should be provided to the objective function f (e.g. data, model objects, etc.)
  import numpy as np
  x=np.array(start)
  xbest=x.copy()
  fbest=1e10
  trace=np.zeros(ndds)
  #get additional arguments
  for counts in range(ndds):
    fval=f(x,*args)
    trace[counts]=fval
    if np.isnan(fval): fval=1e50
    if (fval<fbest):
      fbest=fval
      xbest=x.copy()
    pincl=1-np.log(counts+1)/np.log(ndds)
    npert=0
    #RANDOMLY PERTURBATE PARAMATERS WITH PROBABILITY PINCL, KEEP TRACK OF NO. OF PERTURBATIONS
    x=xbest.copy()
    rno=np.random.uniform(size=len(x))
    select=rno<pincl
    rvar=np.random.normal(size=np.sum(select))
    sig=rdds*(xmax[select]-xmin[select])
    if decrvalrange:
        x[select]=xbest[select]+sig*rvar*pincl
    else:
        x[select]=xbest[select]+sig*rvar
    x=xreflect(x,xmin,xmax)
    npert=np.sum(select)
    #ALWAYS MODIFY AT LEAST ONE PARAMETER, SELECT RANDOMLY IF NONE HAS BEEN MODIFIED
    if (npert==0):
      rno=np.random.uniform(size=1)
      I=int(rno*len(x))
      if (I>len(x)): I=len(x)
      if (I<1): I=1
      rvar=np.random.normal(size=1)
      sig=rdds*(xmax[I]-xmin[I])
      x[I]=xbest[I]+sig*rvar
      x=xreflect(x,xmin,xmax)
  return(xbest,trace)