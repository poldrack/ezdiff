"""
generated simulated data from drift diffusion model
using R rdiffusion package
"""

import numpy
import pandas
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

rtdists = importr('rtdists')

def simulate_ddm(v=1,a=1,t0=0.3,z=0.5,ntrials=500):
    """
    simulate data from drift diffusion model
    """

    rtsim = numpy.array(robjects.r('rdiffusion(%d,a=%f,z=%f,v=%f,t0=%f)'%(ntrials,a,z,v,t0)))
    rtsim[1,:]=rtsim[1,:]-1
    rtdf=pandas.DataFrame(rtsim.T,columns=['rt','response'])

    return(rtdf)
