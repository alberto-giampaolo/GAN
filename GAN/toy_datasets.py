import numpy as np

from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------------------------------------------------------------------
def normalize(x):
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    return x

# ------------------------------------------------------------------------------------------------------------------------------------------
def two_peaks_target(sample_size=10000):
    x = np.random.gamma(1.8,3,(sample_size,1,1)).astype(np.float32)
    x -= x.mean()
    x /= x.std()
    x += np.random.normal(2,0.5,(sample_size,1,1)).astype(np.float32)*(np.random.uniform(0,1,(sample_size,1,1))>0.6).astype(np.float32)
    
    return x

# ------------------------------------------------------------------------------------------------------------------------------------------
def three_peaks_target(sample_size=10000):

    x = two_peaks_target(sample_size)
    x += np.random.normal(-2,0.5,(sample_size,1,1)).astype(np.float32)*(np.random.uniform(0,1,(sample_size,1,1))>0.8).astype(np.float32)
    return x

# ------------------------------------------------------------------------------------------------------------------------------------------
def white_source(sample_size=10000,dim=1,n_sources=1):
    z = np.random.normal(0,1,(sample_size,dim,n_sources))
    return z

# ------------------------------------------------------------------------------------------------------------------------------------------
def unif_source(sample_size=10000,n_sources=1,smear=0.1):
    z = np.random.uniform(-1,1,(sample_size,n_sources,1))
    if smear != None:
        z += np.random.normal(0,0.1,z.shape)
    return z

# ------------------------------------------------------------------------------------------------------------------------------------------
def two_peaks(sample_size=10000,split=True):
    x,z = two_peaks_target(sample_size),white_source(sample_size)
    if split:
        ret = train_test_split(x,z)
    else:
        ret = x,z
    return ret 

# ------------------------------------------------------------------------------------------------------------------------------------------
def three_peaks(sample_size=10000,split=True):
    x,z = three_peaks_target(sample_size),white_source(sample_size)    
    if split:
        ret = train_test_split(x,z)
    else:
        ret = x,z
    return ret


# ------------------------------------------------------------------------------------------------------------------------------------------
def three_peaks_conditional(fc,sample_size=10000,split=True):
    x,z,c = three_peaks_target(sample_size),white_source(sample_size),unif_source(sample_size)
    print(x.shape,z.shape,c.shape)
    cx = np.hstack([c.reshape(-1,1),x.reshape(-1,1)])
    x = normalize(np.apply_along_axis( fc, 1, cx ).reshape(-1,1,1))
    c = normalize(c)
    print(x.shape,z.shape,c.shape)
    if split:
        ret = train_test_split(c,x,z)
    else:
        ret = c,x,z
    return ret


# ------------------------------------------------------------------------------------------------------------------------------------------
def shift_cube(X):
    c,x = X
    val = (2.*c**3 - c**2)
    return  val + x

# ------------------------------------------------------------------------------------------------------------------------------------------
def shift_square(X):
    c,x = X
    val = (c + c**2)
    return  val + x

    
# ------------------------------------------------------------------------------------------------------------------------------------------
def three_peaks_conditional_cube(sample_size=10000,split=True):
    return three_peaks_conditional(shift_cube,sample_size=sample_size)

# ------------------------------------------------------------------------------------------------------------------------------------------
def three_peaks_conditional_square(sample_size=10000,split=True):
    return three_peaks_conditional(shift_square,sample_size=sample_size)

