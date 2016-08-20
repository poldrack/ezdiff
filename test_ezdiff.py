"""
test ezdiff
- use R to get samples from diffusion model
"""



import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import numpy
import pandas
from simulate_ddm import simulate_ddm
from ezdiff import ezdiff


def test_ezdiff_estimates():
    ntrials=250
    nsims=250
    rangerand = lambda x: x[0] + numpy.random.rand()*(x[1]-x[0])
    
    # ranges of parameters 
    a_range=[0.5,1.0] # boundary separation
    t0_range=[0.1,0.5] # non-decision time/response time constant
    v_range=[0.3,2.0]
    v=[]
    a=[]
    t0=[]
    ezest=[]
    rtstats=[]
    for i in range(nsims):
        v.append(rangerand(v_range)) # drift rate
        a.append(rangerand(a_range)) # drift rate
        t0.append(rangerand(t0_range)) # drift rate
    
        rtdf=simulate_ddm(v=v[-1],a=a[-1],t0=t0[-1])
        rtstats.append(rtdf.mean())
        ezest.append(ezdiff(rtdf.rt,rtdf.response))
    
    rtstats=numpy.array(rtstats)
    ezest=numpy.array(ezest)
    acorr=numpy.corrcoef(ezest[:,0],a)[0,1] 
    vcorr=numpy.corrcoef(ezest[:,1],v)[0,1]    
    t0corr=numpy.corrcoef(ezest[:,2],t0)[0,1]    
    
    corrthresh=0.9  # threshold for correlation
    try:
        assert acorr>corrthresh
        assert vcorr>corrthresh
        assert t0corr>corrthresh
    except:
        numpy.savetxt('test_output.txt',numpy.hstack((numpy.vstack((a,v,t0)).T,ezest,rtstats)))
        print(acorr,vcorr,t0corr)
        raise Exception('correlation test failed')
    
    
if __name__=="__main__":
    test_ezdiff_estimates()