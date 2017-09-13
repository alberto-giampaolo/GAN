import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split

zee_data_repo="root://t3dcachedb03.psi.ch//pnfs/psi.ch/cms/trivcat/store/user/musella/zee_data_mc"

def read_hdf(fname,data_repo=None):
    if not os.path.exists(fname) and data_repo != None:
        os.system( "xrdcp %s/%s ." % (data_repo,fname) )
    return pd.read_hdf(fname)
    

def load_zee(version,selection=None):
    if type(version) == tuple or type(version) == list:
        version_data,version_mc = version
    else:
        version_data,version_mc = version,version
    data = read_hdf('data_%s.hd5' % version_data,data_repo=zee_data_repo)
    mc   = read_hdf('mc_%s.hd5' % version_mc,data_repo=zee_data_repo)
    
    if selection:
        data = data.query(selection)
        mc   = mc.query(selection)

    return data,mc
