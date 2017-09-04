from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, Dense, Add, Cropping2D, BatchNormalization, ZeroPadding2D

from keras.optimizers import Adam, RMSprop

from keras.layers import LeakyReLU, Dropout, Activation

## # -------------------------------------------------------------------------------------------------
## def make_gen_unit(name,conv,filters,skip=None,conv_kernel=(1,1),conv_pool=(2,1),deconv_kernel=(1,1),deconv_strides=(2, 1)):
## 
##     if skip != None:
##         up = concatenate( [Conv2DTranspose(filters, deconv_kernel, name="%s_deconv"%name, strides=deconv_strides, padding='same')(conv), skip], axis=3)
##     else:
##         up = conv
## 
##     norm0 = BatchNormalization(name="%s_bn0" % name,momentum=0.9)(up)
##     conv0 = Conv2D(filters,conv_kernel,name="%s_conv0"%name, activation='relu', padding='same')(norm0)
##     # norm1 = BatchNormalization(name="%s_bn1" % name)(conv0)
##     norm1 = conv0
##     last_conv = Conv2D(filters,conv_kernel,name="%s_conv1"%name, activation='relu', padding='same')
##     ret   = last_conv(norm1)
##     skip_ret  = ret
##     
##     if skip == None and conv_pool != None:
##         print(last_conv.output_shape,conv_pool)        
##         if last_conv.output_shape[1] >= conv_pool[0] and last_conv.output_shape[2] >= conv_pool[1]:
##             ## print('aaaaaaa')
##             ret = MaxPooling2D(pool_size=conv_pool,name="%s_pool"%name)(ret)
##     
##     return ret,skip_ret

# -------------------------------------------------------------------------------------------------
def make_gen_unit(name,conv,filters,skip=None,conv_kernel=(1,1),conv_pool=(2,1),deconv_kernel=(1,1),deconv_strides=(2, 1),dropout=None):

    if skip != None:
        up = concatenate( [Conv2DTranspose(filters, deconv_kernel, name="%s_deconv"%name, strides=deconv_strides, padding='same')(conv), skip], axis=3)
    else:
        up = conv

    conv0 = Conv2D(filters,conv_kernel,name="%s_conv0"%name, activation=None, padding='same')(up)
    norm0 = BatchNormalization(name="%s_bn0" % name,momentum=0.9)(conv0)
    last_conv = Activation('relu')
    ret = last_conv(norm0)
    skip_ret  = ret
    
    if skip == None and conv_pool != None:
        print(last_conv.output_shape,conv_pool)        
        if last_conv.output_shape[1] >= conv_pool[0] and last_conv.output_shape[2] >= conv_pool[1]:
            ret = MaxPooling2D(pool_size=conv_pool,name="%s_pool"%name)(ret)

    if dropout != None:
        ret = Dropout(dropout)(ret)
    
    return ret,skip_ret


# -------------------------------------------------------------------------------------------------
def get_generator(input_shape,output_shape,filt_size=[32,64,128,256],sampling_step=(2,1),dropout=None):

    inputs = Input(input_shape)
    ## if len(input_shape) < 2:
    ##     first = Reshape((input_shape[0],1,1),name="reshape_input")(inputs)
    ## else:
    ##     first = inputs

    targets = Cropping2D( cropping=((0, input_shape[0]-output_shape[0]), (0, 0)), name="G_targets" )(inputs)

    conditionals = Cropping2D( cropping=( (output_shape[0],(input_shape[0]-output_shape[0])//2-1), (0,0)), name="G_conditionals" )(inputs)
    
    min_shape = []
    need_padding = False
    for dim,step in enumerate(sampling_step):
        reduction = pow(step,len(sampling_step)+1)
        smallest = max(1,input_shape[dim] // reduction)
        if smallest % 2 != 0 and step != 1:
            smallest += 1
        min_size = smallest * reduction * step
        min_shape.append(min_size)
        print('aaaaaaa',smallest,min_size,input_shape[dim])
        if min_size > input_shape[dim]:
            need_padding = True

    print(need_padding)
    if need_padding:
        paddings = []
        for have,need in zip(input_shape,min_shape):
            paddings.append( (need - have,0) )
        first = ZeroPadding2D( tuple(paddings), name="G_padding" )(inputs)
    else:
        first = inputs
    
    reshaped_targets = Reshape(output_shape,name="G_reshape_targets")(targets)
    
    skip_units = [first]
    conv_units = [first]
    
    for nfilt in filt_size:
        conv,skip = make_gen_unit("G_cunit_%d"%nfilt,conv_units[-1],nfilt,conv_pool=sampling_step)
        conv_units.append(conv)
        skip_units.append(skip)
        
    last_size = filt_size[-1]*2
    last,_ = make_gen_unit("G_cunit_%d" % last_size,conv_units[-1],last_size,conv_pool=None)

    filt_size = reversed(filt_size)
    for nfilt in filt_size:
        skip = skip_units.pop(-1)
        do = None
        if dropout != None:
            do = dropout.pop(0) if type(dropout) != float else dropout
        last,_ = make_gen_unit("G_dunit_%d" % nfilt, last, nfilt, skip,deconv_strides=sampling_step,
                               dropout=do)

    n_outputs = 1
    flat = Flatten(name="G_flat")(last)
    for dim in output_shape:
        n_outputs *= dim
    dense = Dense(n_outputs,name="G_dense",activation="relu")(flat)
    reshaped = Reshape(output_shape,name="G_reshape")(dense)
    predict = Add(name="G_predict")([reshaped,reshaped_targets])
    output = concatenate([predict,conditionals],name="G_output",axis=1)
    
    
    model = Model(inputs=[inputs], outputs=[output])

    return model

# -------------------------------------------------------------------------------------------------
def make_disc_unit(name,inp,filters,dropout=0.4,conv_kernel=(1,1),conv_pool=(2,1)):

    conv = Conv2D(filters,conv_kernel,name="%s_conv"%name, activation=None, padding='same')(inp)
    actv = LeakyReLU(alpha=0.2,name="%s_actv"%name)(conv)
    dout = Dropout(dropout,name="%s_dout"%name)(actv)

    return dout

    

# -------------------------------------------------------------------------------------------------
def get_discriminator(input_shape,filt_size=[32,64,128,256]):

    inputs = Input(input_shape)

    last = inputs
    for nfilts in filt_size:
        last = make_disc_unit("D_conv%d" % nfilts,last,nfilts)
    
    flat = Flatten(name="D_flat")(last)
    dense = Dense(1,name="D_dense",activation="sigmoid")(flat)
    output = dense
    
    model = Model(inputs=[inputs], outputs=[output])
    
    return model

# -------------------------------------------------------------------------------------------------
def discriminator_model(discriminator):
    ## optimizer = RMSprop(lr=0.0002, decay=6e-8)
    optimizer=Adam(lr=1e-4)
    dm = Sequential()
    discriminator.trainable = True
    dm.add(discriminator)
    dm.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return dm

# -------------------------------------------------------------------------------------------------
def adversarial_model(generator,discriminator):
    ## optimizer = RMSprop(lr=0.0002, decay=6e-8)
    optimizer=Adam(lr=1e-4)
    am = Sequential()
    discriminator.trainable = False
    am.add(generator)
    am.add(discriminator)
    am.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return am



### # -------------------------------------------------------------------------------------------------
### def get_unit(name,prev,n_out,dropout=None,activate=False,skip=False,bn=False,nl_activ=False):
### 
###     ## print(n_out)
###     ## name = "%s_%d" % (name,n_out)
###     dense_layer = Dense(n_out,use_bias=True,name="%s_dense" % name)
###     dense = dense_layer(prev)
### 
###     if dropout != None:
###         dense = Dropout(dropout,name="%s_dropout"%name)(dense)
### 
###     if bn:
###         dense = BatchNormalization(name="%s_bn" % name,momentum=.5)(dense)
###     
###     if nl_activ:
###         typ = "sigmoid" if type(nl_activ) != str else nl_activ
###         output_layer = Activation(nl_activ,name="%s_activ"%name)
###     else:
###         ## output_layer = Activation("relu",name="%s_activ"%name)
###         ## output_layer = LeakyReLU(name="%s_activ"%name) 
###         output_layer = PReLU(name="%s_activ"%name)
### 
###     output = output_layer(dense)
###     ## output = dense
### 
###     
###     
###     ### if skip:
###     ###     if dense_layer.output_shape[2] > dense_layer.input_shape[2]:
###     ###         up_layer = UpSampling1D( dense_layer.output_shape[2]/dense_layer.input_shape[2], name="%s_up" %name )
###     ###         up = up_layer(prev)
###     ###         up = Reshape(dense_layer.output_shape,name="%s_up_reshape"%name)(up)
###     ###         print('here',up_layer.input_shape,up_layer.output_shape)
###     ###     else:
###     ###         up = prev
###     ###         if dense_layer.output_shape[2] < dense_layer.input_shape[2]:
###     ###             up_dense_layer = UpSampling1D( dense_layer.input_shape[2]/dense_layer.output_shape[2], name="%s_up" %name )
###     ###             dense = up_dense_layer(dense)
###     ###             
###     ###         dense = Reshape(dense_layer.input_shape[1:],name="%s_up_reshape"%name)(dense)
###     ###         print('there',dense_layer.input_shape,up_dense_layer.input_shape,up_dense_layer.output_shape)
###     ###             
###     ###     dense = Add(name="%s_skip"%name)([output,up])
### 
### 
### 
###     return output# ,output_layer.output_shape
### 
### # -------------------------------------------------------------------------------------------------
### def get_disc_unit(name,prev,n_out,dropout=None):
### 
###     ## print(n_out)
###     ## name = "%s_%d" % (name,n_out)
###     dense = Dense(n_out,use_bias=True,name="%s_dense" % name)(prev)
### 
###     if dropout != None:
###         dense = Dropout(dropout,name="%s_dropout"%name)(dense)
### 
###     output_layer = Activation("relu",name="%s_activ"%name)
###     ## output_layer = Activation("sigmoid",name="%s_activ"%name)
###     ## output_layer = LeakyReLU(name="%s_activ"%name) 
###     ## output_layer = PReLU(name="%s_activ"%name)
###     output = output_layer(dense)
###     # output = dense
###     
###     return output# ,output_layer.output_shape
### 
### 
### # -------------------------------------------------------------------------------------------------
### def get_generator(input_shape,output_shape,kernel_sizes,do_down=True,
###                   do_skip=False,do_poly=False,do_bn=False,do_nl_activ=False):
### 
###     inputs = Input(input_shape,name="G_input")
### 
###     ## powers = [inputs]
###     ## for ip in range(1):
###     ##     powers.append( Multiply(name="G_powers%d" % (ip+2))([inputs,powers[-1]]) )
###     ## cur = concatenate(powers,name="G_powers")
###     cur = inputs
### 
###     ilayer = 1
###     if do_down:
###         for ksize in kernel_sizes:
###             cur = get_unit("G_down%d" % ilayer,cur,ksize,skip=do_skip,bn=do_bn)
###             ilayer += 1
###     
###     for ksize in reversed(kernel_sizes):
###         if do_bn != None:
###             bn = do_bn
###             if type(do_bn) == list:
###                 bn = do_bn.pop(0)
###         if do_nl_activ != None:
###             nl_activ = do_nl_activ
###             if type(do_nl_activ) == list:
###                 nl_activ = do_nl_activ.pop(0)            
###         cur = get_unit("G_up%d" % ilayer,cur,ksize,skip=do_skip,bn=bn,nl_activ=nl_activ)
###         ilayer += 1
### 
###     output_size = 1
###     print(output_shape)
###     for dim in output_shape: output_size *= dim
### 
###     # output = Flatten(name="G_flatten")(cur)
###     output = cur
###     
###     # output = Dense(output_size,activation="relu",use_bias=True,name="G_output")(output)
###     output = Dense(output_size,use_bias=True,name="G_output")(output)
### 
###     if not do_skip and not do_poly:
###         ## output = PReLU(name="G_actviation")(output)
###         output = Add(name="G_add")([inputs,output])
###         
###     if do_poly:
###         terms = []
###         powers = []
###         for iord in range(2):
###             coeff = Dense(output_size,use_bias=True,name="G_output_coef%d" % iord)(output)
###             if len(powers) > 0:
###                 coeff = Multiply(name="G_output_term%d" % iord)([coeff,powers[-1]])
###                 powers.append( Multiply(name="G_powers%d" % (iord+1))([inputs,powers[-1]]) )
###             else:
###                 powers.append(inputs)
###             terms.append(coeff)
###         output = Add(name="G_add")([inputs]+terms)
### 
###     # output = cur
###     output = Reshape(output_shape,name="G_reshape")(output)
###     
###     
###     model = Model(inputs=[inputs],outputs=[output])
###     return model
### 
### # -------------------------------------------------------------------------------------------------
### def get_discriminator(input_shape,kernel_sizes):
### 
###     inputs = Input(input_shape,name="D_input")
### 
###     ilayer = 1
###     cur = inputs
###     for ksize in kernel_sizes:
###         cur = get_disc_unit("D_down%d" % ilayer,cur,ksize)
###         ilayer += 1
###     
###     flat = Flatten(name="D_flat")(cur)
###     output = Dense(1,activation="sigmoid",name="D_output")(flat)
###     
###     model = Model(inputs=[inputs],outputs=[output])
###     return model
### 
### # -------------------------------------------------------------------------------------------------
### def discriminator_model(discriminator):
###     optimizer = RMSprop(lr=0.0002, decay=6e-8)
###     ## optimizer=Adam(lr=1e-5)
###     dm = Sequential()
###     discriminator.trainable = True
###     dm.add(discriminator)
###     dm.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
### 
###     return dm
### 
### # -------------------------------------------------------------------------------------------------
### def adversarial_model(generator,discriminator):
###     optimizer = RMSprop(lr=0.0002, decay=6e-8)
###     ## optimizer=Adam(lr=1e-4)
###     am = Sequential()
###     discriminator.trainable = False
###     am.add(generator)
###     am.add(discriminator)
###     am.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
### 
###     return am
