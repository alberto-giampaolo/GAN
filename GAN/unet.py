import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, UpSampling2D, Flatten, Cropping1D, Convolution2D, Concatenate, Convolution2DTranspose
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout, Lambda
from keras.models import Model, Sequential
import keras.optimizers
from keras.optimizers import Adam, RMSprop

from keras.engine import Layer

from keras.regularizers import l1,l2

from keras import backend as K
import tensorflow as tf

from copy import copy

from .base import Builder, MyGAN, DMBuilder, AMBuilder

from .wgan import WeightClip, wgan_loss

from tqdm import tqdm

# --------------------------------------------------------------------------------------------------
class StochasticThreshold(Layer):

    def __init__(self, **kwargs):
        super(StochasticThreshold, self).__init__(**kwargs)
        ## self.supports_masking = True

    def build(self,input_shape):
        super(StochasticThreshold, self).build(input_shape)
        
    def call(self, inputs):
        inputs,thresholds = inputs
        g = tf.get_default_graph()
        with g.gradient_override_map({"Floor": "Identity"}):
            random_tensor = thresholds + K.random_uniform(shape=K.shape(inputs))
            binary_tensor = tf.floor(random_tensor)
            return inputs*binary_tensor
        
    def compute_output_shape(self,input_shape):
        return input_shape[0]

# --------------------------------------------------------------------------------------------------
class UnetDMBuilder(Builder):

    def __init__(self,optimizer=Adam,loss='binary_crossentropy',loss_weight=None,
                 opt_kwargs=dict(lr=0.0002, decay=6e-8)):
        self.optimizer = optimizer
        if type(self.optimizer) == str:
            self.optimizer = getattr(keras.optimizers,self.optimizer)
        self.opt_kwargs = opt_kwargs
        self.loss = loss
        self.loss_weight = loss_weight
        if type(self.loss) == list:
            self.loss = 2*self.loss
            if self.loss_weight is not None:
                self.loss_weight = 2*self.loss_weight
        super(UnetDMBuilder,self).__init__()

    def build(self,generator,discriminator,do_compile=True):
        
        d_inputs = discriminator.inputs
        y_d = d_inputs[-1]
        g_has_c = False
        if len(d_inputs) == 3:
            g_has_c = True
        g_inputs = generator.inputs
        g_has_noise = False
        if len(g_inputs) == 3:
            g_has_noise = True
            x_g,z_g,c_g = g_inputs
        elif len(g_inputs) == 2:
            x_g,c_g = g_inputs
        else:
            x_g = g_inputs[0]
            
        with K.name_scope("dm"):
            optimizer = self.optimizer(**self.opt_kwargs)
            if do_compile:
                discriminator.trainable = True
                generator.trainable = False
            g_outputs = generator(g_inputs)
            d0_inputs = [x_g]
            d1_inputs = [x_g]
            inputs = [x_g]
            if g_has_noise:
                inputs.append(z_g)
            if g_has_c:
                d0_inputs.append(c_g)
                d1_inputs.append(c_g)
                inputs.append(c_g)
            print(g_outputs[0].get_shape())
            d0_inputs.append(g_outputs[0])
            d1_inputs.append(y_d)
            inputs.append(y_d)
            print(d0_inputs)
            print(d1_inputs)
            d0_outputs = discriminator(d0_inputs)
            d1_outputs = discriminator(d1_inputs)
            if type(d0_outputs) != list:
                d0_outputs = [d0_outputs]
            if type(d1_outputs) != list:
                d1_outputs = [d1_outputs]
            dm = Model(inputs=inputs,outputs=d1_outputs+d0_outputs)
            ## print(dm.summary())
            if do_compile:
                dm.compile(loss=self.loss, loss_weights=self.loss_weight, optimizer=optimizer)
                return dm
            else:
                return dm, optimizer

# --------------------------------------------------------------------------------------------------
class UnetAMBuilder(Builder):

    def __init__(self,optimizer=Adam,loss=None,loss_weight=None,
                 opt_kwargs=dict(lr=0.0002, decay=6e-8)):
        self.optimizer = optimizer
        if type(self.optimizer) == str:
            self.optimizer = getattr(keras.optimizers,self.optimizer)
        self.opt_kwargs = opt_kwargs
        super(UnetAMBuilder,self).__init__()
        self.loss = loss
        self.loss_weight = loss_weight
        
    def build(self,generator,discriminator,do_compile=True):
        optimizer = self.optimizer(**self.opt_kwargs)
        
        d_inputs = discriminator.inputs
        g_has_c = False
        if len(d_inputs) == 3:
            g_has_c = True
        g_inputs = generator.inputs
        g_has_noise = False
        if len(g_inputs) == 3:
            x_g,z_g,c_g = g_inputs
            g_has_noise = True
        elif len(g_inputs) == 2:
            x_g,c_g = g_inputs
        else:
            x_g = g_inputs[0]
            
        with K.name_scope("am"):
            optimizer = self.optimizer(**self.opt_kwargs)
            if do_compile:
                discriminator.trainable = False
                generator.trainable = True
            ## g_outputs = generator(d_inputs)
            inputs = [x_g]
            d_inputs = [x_g]
            if g_has_noise:
                inputs.append(z_g)
            if g_has_c:
                inputs.append(c_g)
                d_inputs.append(c_g)
            print(generator.outputs[0].get_shape())
            d_inputs.append(generator.outputs[0])

            disc = discriminator(d_inputs)
            if type(disc) == list:
                outputs = disc
            else:
                outputs = [disc]
            outputs.append(generator.outputs[-1])
            
            
            am = Model(inputs=inputs,outputs=outputs)
            ## print(am.summary())
            if do_compile:
                am.compile(loss=self.loss, loss_weights=self.loss_weight, optimizer=optimizer)
                return am
            else:
                return am, optimizer

    
# --------------------------------------------------------------------------------------------------
class UnetDBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,name="D",
                 nfilters=32,
                 kernel_size=(3,3),
                 n_layers=4,
                 strides=[1,1,2,1],
                 dropout_rate=0.5,
                 obo_layers=None,
                 obo_nch=4,
                 obo_nch_red=0,
                 obo_nch_min=4,
                 do_total=False,
                 total_layers=[64,64,32,16],
                 use_bias=False,
    ):
        self.name=name
        self.nfilters = nfilters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.obo_layers = obo_layers
        self.obo_nch = obo_nch
        self.obo_nch_red = obo_nch_red
        self.obo_nch_min = obo_nch_min
        assert( len(strides) == n_layers )
        self.do_total = do_total
        self.total_layers = total_layers
        self.use_bias = use_bias
        super(UnetDBuilder,self).__init__()
        
    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,c_shape=None):

        input_shape = x_shape
        x_inputs = Input(input_shape,name="%s_x_input" % self.name)
        y_inputs = Input(input_shape,name="%s_y_input" % self.name)

        concat_inputs = Concatenate(name="%s_concat_input" % self.name,axis=-1)([x_inputs,y_inputs])
        inputs = [x_inputs,y_inputs]

        if c_shape is not None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            inputs = [x_inputs,c_inputs,y_inputs]
            
        wdout  = concat_inputs
        wodout = concat_inputs
        nfilters = self.nfilters*2
        for il,stride in enumerate(self.strides):
            conv = Convolution2D( nfilters, kernel_size=self.kernel_size,
                                  use_bias=self.use_bias,
                                  name="%s_conv%d" % (self.name, il),
                                  strides=(stride,stride), padding='valid' )
            if il > 0:
                bn = BatchNormalization(axis=1,name="%s_bn%d" % (self.name,il) )
                wdout  = bn(wdout)
                wodout = bn(wodout)
            do = Dropout(self.dropout_rate, name="%s_do%d" % (self.name, il) )
            act = LeakyReLU(0.2,name="%s_act%d" % ( self.name, il ) )
            wdout  = act( do( conv( wdout  ) ) )
            wodout = act(     conv( wodout )   )
            nfilters *= 2


        if c_shape is not None:
            nembed = nfilters // 2
            c_bn0 = BatchNormalization(axis=-1, name="%s_c_bn0" % self.name)(c_inputs)
            c_dense = Dense(nembed, use_bias=self.use_bias, name="%s_c_dense" % self.name)(c_bn0)
            c_bn = BatchNormalization(axis=-1, name="%s_c_bn" % self.name)(c_dense)
            c_rshp = Reshape( (1, 1, nembed), name="%s_c_rshp" % self.name)(c_bn)
            ups = wdout.get_shape()
            ups = ( int(ups[1]), int(ups[2]) )
            c_up = UpSampling2D( size=ups, name="%s_c_up" % self.name)(c_rshp)
            
            concat = Concatenate(name="%s_concat" % self.name,axis=-1)
            wdout = concat([wdout,c_up])
            wodout = concat([wodout,c_up])

            
        if self.obo_layers is not None:
            nch = self.obo_nch
            if nch == 0:
                nch = nfilters // 2
            for il in range(self.obo_layers):
                fc = Convolution2D(nch,kernel_size=(1,1),
                                   use_bias=self.use_bias,
                                   strides=(1,1),name="%s_fc%d" % (self.name,il) )
                wdout = fc(wdout)
                wodout = fc(wodout)
                
                bn = BatchNormalization(axis=1,name="%s_obo_bn%d" % (self.name,il) )
                wdout  = bn(wdout)
                wodout = bn(wodout)
                
                act = LeakyReLU(0.2,name="%s_obo_act%d" % ( self.name, il ) )
                wdout  = act(wdout)
                wodout = act(wodout)

                if self.obo_nch_red != 0:
                    nch = max(self.obo_nch_min,nch//self.obo_nch_red)
                
        flat = Flatten(name="%s_flat" % self.name)
        wdout = flat(wdout)
        wodout = flat(wodout)

        
        dense = Dense(1,name="%s_dense" % self.name)
        wdout = dense(wdout)
        wodout = dense(wodout)
        
        out = Activation("sigmoid",name="%s_out" % self.name)
        wdout = out(wdout)
        wodout = out(wodout)

        wdooutputs = [wdout]
        wodooutputs = [wodout]

        if self.do_total:
            sum_x = Lambda(lambda x: K.sum(x,axis=(1,2)),#output_shape=(1,),
                           name="%s_sum_x"%self.name)(x_inputs)
            sum_y = Lambda(lambda x: K.sum(x,axis=(1,2)),#output_shape=(1,),
                           name="%s_sum_y"%self.name)(y_inputs)
            concat_sum = [sum_x,sum_y]
            print(sum_x.get_shape(),sum_y.get_shape())
            if c_shape is not None:
                c_flat = Flatten(name="%s_c_flt" % self.name)(c_bn)
                concat_sum.append(c_flat)
            concat_sum = Concatenate(name="%s_concat_sum" % self.name,axis=-1)(concat_sum)
            wdout = concat_sum
            wodout = concat_sum
            for il,nc in enumerate(self.total_layers):
                t_dense = Dense(nembed, use_bias=self.use_bias,
                                name="%s_t_dense%d" % ( self.name, il ) )
                t_bn = BatchNormalization(axis=-1, name="%s_t_bn%d" % ( self.name, il ) )
                t_do = Dropout(self.dropout_rate, name="%s_t_do%d" % ( self.name, il ) )
                t_act = LeakyReLU(0.2,name="%s_t_act%d" % ( self.name, il ) )
                wdout  = t_act( t_do( t_bn( t_dense( wdout  ) ) ) )
                wodout = t_act(       t_bn( t_dense( wodout )   ) )

            dense = Dense(1,name="%s_t_dense" % self.name)
            wdout = dense(wdout)
            wodout = dense(wodout)
        
            out = Activation("sigmoid",name="%s_t_out" % self.name)
            wdout = out(wdout)
            wodout = out(wodout)
                        
            wdooutputs.append(wdout)
            wodooutputs.append(wodout)

                
        model = (Model(inputs=inputs,outputs=wdooutputs),Model(inputs=inputs,outputs=wodooutputs))
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
                 stochastic_layer=False,
                 soft_mask=False,
                 use_bias=False,
                 ):
        self.name = name
        self.nfilters = nfilters
        self.encoding_strides = (encoding_stride,encoding_stride)
        self.stochastic_layer=stochastic_layer
        self.soft_mask=soft_mask
        
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
        self.use_bias = use_bias
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
            z_dense = Dense(nembed, use_bias=self.use_bias,
                            name="%s_z_dense" % self.name)(z_inputs)
            z_bn = BatchNormalization(axis=-1, name="%s_z_bn" % self.name)(z_dense)
            z_rshp = Reshape( (1, 1, self.nch), name="%s_z_rshp" % self.name)(z_bn)
            
        ## Conditionals embedding
        if c_shape is not None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            inputs.append(c_inputs)
            c_bn0 = BatchNormalization(axis=-1, name="%s_c_bn0" % self.name)(c_inputs)
            c_dense = Dense(nembed, use_bias=self.use_bias,
                            name="%s_c_dense" % self.name)(c_bn0)
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
                                 use_bias=self.use_bias,
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
            skip = skips.pop(-1)
            strides = self.decoding_strides
            if len(skips) > 0: # adjust strides to match dimension of next skip connection
                if skips[-1][0].shape[1] == cur.shape[1]:
                    strides = (1,1)
            if il > 0: # add skip connection
                skip.append(cur)
                cur = Concatenate(name="%s_dec_concat%d" % (self.name, il) )(skip)
            cur = Convolution2DTranspose(layer,
                                         kernel_size=self.decoding_filter,
                                         strides=strides,
                                         use_bias=self.use_bias,
                                         padding="same",
                                         name="%s_dec_deconv%d" % (self.name,il) )(cur)
            cur = BatchNormalization(name="%s_dec_bn%d" % (self.name,il) )(cur)
            if self.decoding_dropout[il] > 0.:
                cur = Dropout(self.decoding_dropout[il], name="%s_dec_do%d" % (self.name, il) )(cur)
                
            cur = LeakyReLU(0.2,name="%s_dec_act%d" % ( self.name, il ) )(cur)

        ## output layer
        cur = Concatenate(name="%s_out_concat%d" % (self.name, il) )([x_inputs,cur])
        unet_out = cur
        
        cur = Convolution2DTranspose(x_shape[-1],kernel_size=self.decoding_filter,
                                     use_bias=self.use_bias,
                                     name="%s_out_deconv" % self.name, padding='same' )(cur)
        out = Activation(self.activation,name="%s_out_actv" % self.name)(cur) ## output is > 0
        outputs = [out]

        if self.soft_mask or self.stochastic_layer:
            cur = Convolution2DTranspose(x_shape[-1],kernel_size=self.decoding_filter,
                                         use_bias=self.use_bias,
                                         name="%s_out_sm_deconv" % self.name, padding='same' )(unet_out)
            soft_mask = Activation("sigmoid",name="%s_out_sm" % self.name)(cur)
            
        if self.stochastic_layer:
            stoc_out = StochasticThreshold(name="%s_stochastic_out" % self.name)([out,soft_mask])
            outputs = [stoc_out]
            
        if self.soft_mask:
            ## outputs.append(soft_mask)
            concat = Concatenate(name="%s_out_concat" % (self.name), axis=-1 )([out,soft_mask])
            if self.stochastic_layer:
                ## concat = Concatenate(name="%s_out_concat" % (self.name), axis=-1 )([out,soft_mask,stoc_out])
                outputs = [stoc_out,concat]
            else:
                ## concat = Concatenate(name="%s_out_concat" % (self.name), axis=-1 )([out,soft_mask])
                outputs = [concat]
            
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
        
        dmBuilder = UnetDMBuilder(**dm_opts)
        amBuilder = UnetAMBuilder(**am_opts)
        
        super(MyUnetGAN,self).__init__(x_shape,z_shape,gBuilder,dBuilder,dmBuilder,amBuilder,**kwargs)

    def compile(self):
        self.dm = self.dmBuilder(self.get_generator(),self.get_discriminator()[0])
        self.am = self.amBuilder(self.get_generator(),self.get_discriminator()[1])
        
        return self.am,self.dm

    def fit_generator(self,train_generator,epochs,steps_per_epoch,
                      validation_data=None,validation_steps=None,
                      monitor_dir=None
		      ## callbacks=[csv,checkpoint],
    ):
        from collections import OrderedDict
        
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        
        if monitor_dir is not None:
            generator.save("%s/model.hdf5" % monitor_dir )
            discriminator[0].save("%s/discriminator.hdf5" % monitor_dir )
           
        for iepoch in tqdm(range(epochs),"epochs"):
            prog_bar = tqdm(range(steps_per_epoch),"epoch %d" % iepoch,leave=True)
            for ibatch in prog_bar:
                batch = next(train_generator)

                x_batch, y_batch = batch[:2]
                w_batch = None
                if len(batch) == 3:
                    w_batch = [ batch[2], batch[2] ]
                g_w = w_batch
                d_w = w_batch
                    
                d_x = x_batch+[y_batch[:,:,:,:1]]
                g_x = x_batch

                ones = np.ones( (x_batch[0].shape[0],1) )
                zeros = np.zeros( (x_batch[0].shape[0],1) )
                if len(self.amBuilder.loss) == 2:
                    d_y = [ ones, zeros   ]  
                    g_y = [ ones, y_batch ]
                else:
                    d_y = [ ones, ones, zeros, zeros  ]
                    g_y = [ ones, ones, y_batch ]
                    if w_batch is not None:
                        d_w = [ w_batch[0] ] * 4
                        g_w = [ w_batch[0] ] * 3
                        
                generator.trainable = False
                discriminator[0].trainable = True
                discriminator[1].trainable = True
                d_loss = self.dm.train_on_batch(d_x,d_y,sample_weight=d_w)
                generator.trainable = True
                discriminator[0].trainable = False
                discriminator[1].trainable = False
                a_loss = self.am.train_on_batch(g_x,g_y,sample_weight=g_w)

                postfix = OrderedDict([("a_loss",a_loss), ("d_loss",d_loss),])
                # postfix = dict(d_loss=d_loss, a_loss=a_loss)
                prog_bar.set_postfix( postfix )

            if monitor_dir is not None:
                generator.save("%s/model-%01d.hdf5" % (monitor_dir,iepoch) )
                discriminator[0].save("%s/discriminator-%01d.hdf5" % (monitor_dir,iepoch) )
            
            ### if validation_data is not None:
            ###     for ival in range(validation_steps):
                        
