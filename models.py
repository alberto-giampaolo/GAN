from __future__ import print_function

from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten, Dense, Add, Cropping2D, BatchNormalization, ZeroPadding2D

from keras.optimizers import Adam, RMSprop

from keras.layers import LeakyReLU, Dropout

# -------------------------------------------------------------------------------------------------
def make_gen_unit(name,conv,filters,skip=None,conv_kernel=(1,1),conv_pool=(2,1),deconv_kernel=(1,1),deconv_strides=(2, 1)):

    if skip != None:
        up = concatenate( [Conv2DTranspose(filters, deconv_kernel, name="%s_deconv"%name, strides=deconv_strides, padding='same')(conv), skip], axis=3)
    else:
        up = conv

    norm0 = BatchNormalization(name="%s_bn0" % name,momentum=0.9)(up)
    conv0 = Conv2D(filters,conv_kernel,name="%s_conv0"%name, activation='relu', padding='same')(norm0)
    # norm1 = BatchNormalization(name="%s_bn1" % name)(conv0)
    norm1 = conv0
    last_conv = Conv2D(filters,conv_kernel,name="%s_conv1"%name, activation='relu', padding='same')
    ret   = last_conv(norm1)
    skip_ret  = ret
    
    if skip == None and conv_pool != None:
        print(last_conv.output_shape,conv_pool)        
        if last_conv.output_shape[1] >= conv_pool[0] and last_conv.output_shape[2] >= conv_pool[1]:
            ## print('aaaaaaa')
            ret = MaxPooling2D(pool_size=conv_pool,name="%s_pool"%name)(ret)
    
    return ret,skip_ret
        

# -------------------------------------------------------------------------------------------------
def get_generator(input_shape,output_shape,filt_size=[32,64,128,256],sampling_step=(2,1)):

    inputs = Input(input_shape)
    ## if len(input_shape) < 2:
    ##     first = Reshape((input_shape[0],1,1),name="reshape_input")(inputs)
    ## else:
    ##     first = inputs

    targets = Cropping2D( cropping=((0, input_shape[0]-output_shape[0]), (0, 0)), name="G_targets" )(inputs)
    
    min_shape = []
    need_padding = False
    for dim,step in enumerate(sampling_step):
        reduction = pow(step,len(sampling_step)+1)
        smallest = input_shape[dim] // reduction
        if smallest % 2 != 0 and step != 1:
            smallest += 1
        min_size = smallest * reduction * step
        min_shape.append(min_size)
        print(smallest,min_size,input_shape[dim])
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
        last,_ = make_gen_unit("G_dunit_%d" % nfilt, last, nfilt, skip,deconv_strides=sampling_step)

    n_outputs = 1
    flat = Flatten(name="G_flat")(last)
    for dim in output_shape:
        n_outputs *= dim
    dense = Dense(n_outputs,name="G_dense",activation="relu")(flat)
    reshaped = Reshape(output_shape,name="G_reshape")(dense)
    output = Add(name="G_output")([reshaped,reshaped_targets])
    
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

