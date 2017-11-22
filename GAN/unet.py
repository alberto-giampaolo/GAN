import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, Cropping1D, Convolution2D, Concatenate, Convolution2DTranspose
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
import keras.optimizers
from keras.optimizers import Adam, RMSprop

from keras.regularizers import l1,l2

from keras import backend as K

from copy import copy

from .base import Builder, MyGAN, DMBuilder, AMBuilder

from .wgan import WeightClip, wgan_loss


# --------------------------------------------------------------------------------------------------
class UnetDBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,name="D",nch=256,dropout_rate=0.5):
        self.nch = nch
        self.nch2 = nch // 2
        self.dropout_rate = dropout_rate
        super(UnetDBuilder,self).__init__()
        
    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,c_shape=None):

        input_shape = x_shape
        inputs = Input(input_shape,name="%s_input" % self.name)
        
        cur = Convolution2D(self.nch2, subsample=(2, 2), border_mode='same', activation='relu', name="%s_conv1" % self.name)(inputs)
        cur = LeakyReLU(0.2,name="%s_actv1" % self.name)(cur)
        wodout = cur
        wdout = Dropout(self.dropout_rate,name="%s_dout1" % self.name)(cur)
        
        conv2 = Convolution2D(self.nch, 5, 5, subsample=(2, 2), name="%s_conv2" % self.name,
                              border_mode='same', activation='relu')
        wdout = conv2(wdout)
        wodout = conv2(wodout)
        acvt2 = LeakyReLU(0.2, name="%s_actv2" % self.name)
        wdout = acvt2(wdout)
        wodout = acvt2(wodout)

        dout2 = Dropout(self.dropout_rate, name="%s_dout2" % self.name)
        wdout = dout2(wdout)

        flat = Flatten(name="%s_flat" % self.name)
        wdout = flat(wdout)
        wodout = flat(wodout)
        dense = Dense(self.nch2,name="%s_dense" % self.name)
        wdout = dense(wdout)
        wodout = dense(wodout)
        actv3 = LeakyReLU(0.2,name="%s_actv3" % self.name)
        wdout = actv3(wdout)
        wodout = actv3(wodout)
        dout3 = Dropout(self.dropout_rate,name="%s_dout3" % self.name)
        wdout = dout3(wdout)
                
        outputs = [wdout,wodout]
        model = (Model(inputs=inputs,outputs=[output[0]]),Model(inputs=inputs,outputs=[output[1]]))
        return model

# --------------------------------------------------------------------------------------------------
class UnetGBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,name="G",
                 nfilters=32,
                 min_encoding=2, max_encoding=8, n_encoding=6,
                 encoding_filter=(3,3),
                 encoding_stride=2, decoding_last_plateau=2,
                 decoding_last_dropout=2,
                 decoding_dropout=0.5,
                 decoding_filter=(3,3),
                 decoding_stride=2,
                 activation="relu",
                 ):
        self.name = name
        self.nfilters = nfilters
        self.encoding_strides = (encoding_stride,encoding_stride)

        self.encoding_layers = []
        print('encoding')
        iencoding = min_encoding
        for ienc in range(n_encoding):
            print(iencoding)
            self.encoding_layers.append( self.nfilters * iencoding )
            iencoding = min(max_encoding,iencoding*encoding_stride)
        self.encoding_filter = encoding_filter

        self.decoding_strides = (decoding_stride,decoding_stride)
        self.decoding_layers = []
        self.decoding_dropout = []
        idecoding = iencoding
        print('decoding')
        for idec in range(n_encoding-1):
            print(idecoding)
            self.decoding_layers.append( self.nfilters * idecoding )
            if idec < decoding_last_dropout:
                self.decoding_dropout.append( decoding_dropout )
            else:
                self.decoding_dropout.append( 0. )
            if idec+1 >= decoding_last_plateau:
                idecoding = max(1,idecoding//decoding_stride)
        self.decoding_filter = decoding_filter

        self.activation = activation
        
        ## embedding for c and z has the same size as the last encoding layer
        self.nch = self.encoding_layers[0] 
        super(UnetGBuilder,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,z_shape=None,c_shape=None):

        x_inputs = Input(x_shape,name="%s_x_input" % self.name)
        inputs = [x_inputs]
        nembed = self.nch

        ## Noise embedding
        if z_shape is not None:
            z_inputs = Input(z_shape,name="%s_z_input" % self.name)
            inputs.append(z_inputs)
            z_dense = Dense(nembed, name="%s_z_dense" % self.name)(z_inputs)
            z_bn = BatchNormalization(axis=-1, name="%s_z_bn" % self.name)(z_dense)
            z_rshp = Reshape( (1, 1, self.nch), name="%s_z_rshp" % self.name)(z_bn)
            
        ## Conditionals embedding
        if c_shape is not None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            inputs.append(c_inputs)
            c_bn0 = BatchNormalization(axis=-1, name="%s_c_bn0" % self.name)(c_inputs)
            c_dense = Dense(nembed, name="%s_c_dense" % self.name)(c_bn0)
            c_bn = BatchNormalization(axis=-1, name="%s_c_bn" % self.name)(c_dense)
            c_rshp = Reshape( (1, 1, self.nch), name="%s_c_rshp" % self.name)(c_bn)
            
        cur = x_inputs
        skips = []
        ## Do encoding.
        ## At each step axis 1 and 2 are divided by a factor stride, axis 3 dimensionality
        ## increases according to input.
        for il,layer in enumerate(self.encoding_layers):
            ## print('encoding inp ', il, cur.shape, layer)
            cur = Convolution2D( layer, kernel_size=self.encoding_filter,
                                        name="%s_enc_conv%d" % (self.name, il),
                                        strides=self.encoding_strides, padding='same' )(cur)
            cur = BatchNormalization(axis=1,name="%s_enc_bn%d" % (self.name,il) )(cur)
            skips.append( [cur] )
            ## print('encoding out ', il, cur.shape, layer)
            cur = LeakyReLU(0.2,name="%s_enc_act%d" % ( self.name, il ) )(cur)
            
        ## inject z and c here
        last_skip = []
        if z_shape is not None:
            last_skip.append(z_rshp)
        if c_shape is not None:
            last_skip.append(c_rshp)
        if len(last_skip) > 0:
            last_skip.append(cur)
            cur = Concatenate(name="%s_dec_concat%d" % (self.name, il) )(last_skip)
        skips.pop(-1)
        
        ## Do decoding
        last_layer = len(self.encoding_layers) - 1
        for il, layer in enumerate(self.decoding_layers):
            ## print('decoding ', il, layer)
            skip = skips.pop(-1)
            ## print('decoding inp ', il, cur.shape, skip[-1].shape, layer)
            out_shape = (cur.shape[1]*2,cur.shape[2]*2)
            if il > 0: # add skip connection
                skip.append(cur)
                cur = Concatenate(name="%s_dec_concat%d" % (self.name, il) )(skip)
            ## print('decoding concat ', il, cur.shape)
            cur = Convolution2DTranspose(layer,
                                         kernel_size=self.decoding_filter,
                                         strides=self.decoding_strides,
                                         padding="same",
                                         name="%s_dec_deconv%d" % (self.name,il) )(cur)
            ## print('decoding transp ', il, cur.shape)
            ## print('decoding bn ', il, cur.shape)
            cur = BatchNormalization(name="%s_dec_bn%d" % (self.name,il) )(cur)
            if self.decoding_dropout[il] > 0.:
                ## print('decoding do ', il, cur.shape)
                cur = Dropout(self.decoding_dropout[il], name="%s_dec_do%d" % (self.name, il) )(cur)
                
            ## print('decoding lr ', il, cur.shape)
            cur = LeakyReLU(0.2,name="%s_dec_act%d" % ( self.name, il ) )(cur)
            ## print('decoding out ', il, cur.shape)

        ## output layer
        cur = Concatenate(name="%s_out_concat%d" % (self.name, il) )([x_inputs,cur])
        cur = Convolution2DTranspose(x_shape[-1],kernel_size=self.decoding_filter,
                                     name="%s_out_deconv" % self.name, padding='same' )(cur)
        out = Activation(self.activation,name="%s_out_actv" % self.name)(cur) ## output is > 0
        outputs = [out]
        
        model = Model(inputs=inputs,outputs=outputs)
        return model
        
# --------------------------------------------------------------------------------------------------
class MyUnetGAN(MyGAN):

    def __init__(self,x_shape,z_shape,
                 g_opts=dict(),
                 d_opts=dict(),
                 dm_opts=dict(),
                 am_opts=dict(),
                 **kwargs
    ):

        gBuilder = UnetGBuilder(**g_opts)
        dBuilder = UnetDBuilder(**d_opts)
        
        dmBuilder = DMBuilder(**dm_opts)
        amBuilder = AMBuilder(**am_opts)
        
        super(MyUnetGAN,self).__init__(x_shape,z_shape,gBuilder,dBuilder,dmBuilder,amBuilder,**kwargs)
    
