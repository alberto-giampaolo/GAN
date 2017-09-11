from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer


default_tranformers = dict(minmax=RobustScaler(quantile_range=(10.,90.)),
                           gaus=QuantileTransformer(output_distribution="normal"))

# ------------------------------------------------------------------------------------------------
def transform(data_x,data_c,mc_x,mc_c,transform='minmax',reshape=True,return_scalers=True):
    from sklearn.base import clone
    
    if type(transform) == str:
        transform = default_tranformers[transform]
    
    scaler_x = clone(transform)
    scaler_c = clone(transform)

    mc_x = scaler_x.fit_transform(mc_x)
    data_x = scaler_x.transform(data_x)
    
    mc_c = scaler_c.fit_transform(mc_c)
    data_c = scaler_c.transform(data_c)

    if reshape:
        mc_x = mc_x.reshape(-1,1,mc_x.shape[-1])
        mc_c = mc_c.reshape(-1,1,mc_c.shape[-1])
        data_x = data_x.reshape(-1,1,data_x.shape[-1])
        data_c = data_c.reshape(-1,1,data_c.shape[-1])

    if return_scalers:
        return data_x,data_c,mc_x,mc_c,scaler_x,scaler_c

    return data_x,data_c,mc_x,mc_c
