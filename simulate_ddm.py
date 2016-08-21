"""
generated simulated data from drift diffusion model
using R rdiffusion package
"""

import os
import numpy
import pandas
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

rtdists = importr('rtdists')

def simulate_ddm(v=1,a=1,t0=0.3,z=0.5,ntrials=1000):
    """
    simulate data from drift diffusion model
    v = drift rate
    a = boundary separation
    t0 = non-decision time
    """
    assert ntrials > 0
    assert a > 0
    assert t0 > 0
    
    rtsim = numpy.array(robjects.r('rdiffusion(%d,a=%f,z=%f,v=%f,t0=%f)'%(ntrials,a,z,v,t0)))
    rtsim[1,:]=rtsim[1,:]-1  # simulation returns 1/2 for response, change it to 0/1
    rtdf=pandas.DataFrame(rtsim.T,columns=['rt','response'])
    rtdf['v']=v
    rtdf['a']=a
    rtdf['t0']=t0

    return(rtdf)

def mk_simulated_data(a_range=[0.5,1.0],t0_range=[0.1,0.5],v_range=[0.3,2.0],
                      nruns=1000,ntrials=1000,dataloc='./simulated_data'):
    rangerand = lambda x: x[0] + numpy.random.rand()*(x[1]-x[0])
    v=[]
    a=[]
    t0=[]

    if not os.path.exists(dataloc):
        os.mkdir(dataloc)
    
    for i in range(nruns):
        v.append(rangerand(v_range)) # drift rate
        a.append(rangerand(a_range)) # drift rate
        t0.append(rangerand(t0_range)) # drift rate
    
        rtdf=simulate_ddm(v=v[-1],a=a[-1],t0=t0[-1],ntrials=ntrials)
        rtdf.to_csv(os.path.join(dataloc,'simdata_run%05d.csv'%i))

if __name__=='__main__':
    mk_simulated_data()