import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def load_zee(version,selection=None):
    if type(version) == tuple or type(version) == list:
        version_data,version_mc = version
    else:
        version_data,version_mc = version,version
    data = pd.read_hdf('data_%s.hd5' % version_data)
    mc   = pd.read_hdf('mc_%s.hd5' % version_mc)
    
    if selection:
        data = data.query(selection)
        mc   = mc.query(selection)

    return data,mc
