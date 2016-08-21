"""
test ezdiff
- use R to get samples from diffusion model
"""


import glob
import numpy
import pandas
from simulate_ddm import mk_simulated_data
from ezdiff import ezdiff


def test_ezdiff_on_canned_data():
    datafiles=glob.glob('simulated_data/simdata*csv')
    if len(datafiles)==0:
        mk_simulated_data()
        datafiles=glob.glob('simulated_data/simdata*csv')
    datafiles.sort()
    assert len(datafiles)>1
    
    ezd_est=numpy.zeros((len(datafiles),6))
    
    for i,f in enumerate(datafiles):
        df=pandas.read_csv(f)
        ezd_est[i,3:]=ezdiff(df.rt,df.response)
        ezd_est[i,:3]=[df.a.iloc[0],df.v.iloc[0],df.t0.iloc[0]]
        
    ezdiff_est=pandas.DataFrame(data=ezd_est,columns=['a','v','t0','a_est','v_est','t0_est'])
    corr_thresh=0.9  # threshold for correlation
    assert ezdiff_est.corr().v_est.v > corr_thresh
    assert ezdiff_est.corr().a_est.a > corr_thresh
    assert ezdiff_est.corr().t0_est.t0 > corr_thresh
    
