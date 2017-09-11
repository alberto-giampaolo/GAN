import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

from copy import copy

from base import Builder, MyGAN

# --------------------------------------------------------------------------------------------------
class FFDBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,kernel_sizes,name="D"):
        self.kernel_sizes = kernel_sizes
        self.name = name
        super(FFDBuilder,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,c_shape=None):
        input_shape = x_shape
        
        inputs = Input(input_shape,name="%s_input" % self.name)
        if c_shape != None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            cur = concatenate( [c_inputs,inputs], name = "%s_all_inputs" % self.name )
            ### cur = Reshape((2,1))(cur)
            ### cur = Cropping1D( cropping=(1,0) )(cur)
            inputs = [c_inputs,inputs]
        else:
            cur = inputs
            inputs = [inputs]
        ilayer = 1
        for ksize in self.kernel_sizes:
            cur = self.get_unit("%s_down%d" % (self.name,ilayer),cur,ksize)
            ilayer += 1
            
        flat = Flatten(name="%s_flat" % self.name)(cur)
        output = Dense(1,activation="sigmoid",name="%s_output" % self.name)(flat)
            
        model = Model(inputs=inputs,outputs=[output])
        return model

    # --------------------------------------------------------------------------------------------------
    def get_unit(self,name,prev,n_out,dropout=None):

        dense = Dense(n_out,use_bias=True,name="%s_dense" % name)(prev)
        
        if dropout != None:
            dense = Dropout(dropout,name="%s_dropout"%name)(dense)
            
        output_layer = Activation("relu",name="%s_activ"%name)
        ## output_layer = Activation("tanh",name="%s_activ"%name)
        ## output_layer = LeakyReLU(name="%s_activ"%name) 
        ## output_layer = PReLU(name="%s_activ"%name)
        output = output_layer(dense)
        
        return output


# --------------------------------------------------------------------------------------------------
class FFGBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,kernel_sizes,do_down=False,do_skip=False,do_poly=False,do_bn=False,
                 do_nl_activ=False,name="G"):
        self.kernel_sizes = kernel_sizes
        self.do_down = do_down
        self.do_skip = do_skip
        self.do_poly = do_poly
        self.do_bn = do_bn
        self.do_nl_activ = do_nl_activ
        self.name = name
        super(FFGBuilder,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,z_shape,c_shape=None):
        
        do_down = copy(self.do_down)
        do_skip = copy(self.do_skip)
        do_poly = copy(self.do_poly)
        do_bn = copy(self.do_bn)
        do_nl_activ = copy(self.do_nl_activ)
        input_shape = z_shape
        output_shape = x_shape
        
        x_inputs = Input(input_shape,name="%s_input" % self.name)
        if c_shape != None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            cur = concatenate( [c_inputs,x_inputs], name = "%s_all_inputs" % self.name)
            ## cur = Reshape((2,1))(cur)
            ## cur = Cropping1D( cropping=(1,0) )(cur)
            inputs = [c_inputs,x_inputs]
            outputs = [c_inputs]
        else:
            cur = x_inputs
            inputs = [x_inputs]
            outputs = []
        
        ## powers = [inputs]
        ## for ip in range(1):
        ##     powers.append( Multiply(name="%s_powers%d" % (self.name,ip+2))([inputs,powers[-1]]) )
        ## cur = concatenate(powers,name="%s_powers" % self.name)
        ## cur = inputs

        ilayer = 1
        if do_down:
            for ksize in self.kernel_sizes:
                cur = self.get_unit("%s_down%d" % (ilayer,self.name),cur,ksize,skip=do_skip,bn=do_bn)
                ilayer += 1
                
        for ksize in reversed(self.kernel_sizes):
            if do_bn != None:
                bn = do_bn
                if type(do_bn) == list:
                    bn = do_bn.pop(0)
            if do_nl_activ != None:
                nl_activ = do_nl_activ
            if type(do_nl_activ) == list:
                nl_activ = do_nl_activ.pop(0)            
            cur = self.get_unit("%s_up%d" % (self.name,ilayer),cur,ksize,skip=do_skip,bn=bn,nl_activ=nl_activ)
            ilayer += 1

        output_size = 1
        print(output_shape)
        for dim in output_shape: output_size *= dim

        # output = Flatten(name="%s_flatten" % self.name)(cur)
        output = cur
        
        # output = Dense(output_size,activation="relu",use_bias=True,name="%s_output" % self.name)(output)
        output = Dense(output_size,use_bias=True,name="%s_output" % self.name)(output)

        if not do_skip and not do_poly:
            ## output = PReLU(name="%s_actviation" % self.name)(output)
            output = Add(name="%s_add" % self.name)([x_inputs,output])
        
        if do_poly:
            terms = []
            powers = []
            for iord in range(2):
                coeff = Dense(output_size,use_bias=True,name="%s_output_coef%d" % (self.name,iord))(output)
                if len(powers) > 0:
                    coeff = Multiply(name="%s_output_term%d" % (self.name,iord) )([coeff,powers[-1]])
                    powers.append( Multiply(name="%s_powers%d" % (self.name,iord+1))([x_inputs,powers[-1]]) )
                else:
                    powers.append(x_inputs)
            terms.append(coeff)
            output = Add(name="%s_add" % self.name)([x_inputs]+terms)

        # output = cur
        # output = Reshape(output_shape,name="%s_reshape" % self.name)(output)
        outputs.append(output)
        
        model = Model(inputs=inputs,outputs=outputs)
        return model
    
    # --------------------------------------------------------------------------------------------------
    def get_unit(self,name,prev,n_out,dropout=None,activate=False,skip=False,bn=False,nl_activ=False):

        inp = prev        
        if bn:
            prev = BatchNormalization(name="%s_bn" % name,momentum=.5)(prev)
    
        if dropout != None:
            prev = Dropout(dropout,name="%s_dropout"%name)(prev)

        dense_layer = Dense(n_out,use_bias=True,name="%s_dense" % name)
        dense = dense_layer(prev)

        if nl_activ != False:
            typ = "sigmoid" if type(nl_activ) != str else nl_activ
            output_layer = Activation(typ,name="%s_activ"%name)
        else:
            ## output_layer = Activation("relu",name="%s_activ"%name)
            ## output_layer = LeakyReLU(name="%s_activ"%name) 
            output_layer = PReLU(name="%s_activ"%name)
        output = output_layer(dense)

        if skip:
            if dense_layer.output_shape[2] > dense_layer.input_shape[2]:
                up_layer = UpSampling1D( dense_layer.output_shape[2]/dense_layer.input_shape[2], name="%s_up" %name )
                up = up_layer(inp)
                up = Reshape(dense_layer.output_shape[1:],name="%s_up_reshape"%name)(up)
                print(up_layer.input_shape,up_layer.output_shape)
            else:
                up = inp
                if dense_layer.output_shape[2] < dense_layer.input_shape[2]:
                    up_dense_layer = UpSampling1D( dense_layer.input_shape[2]/dense_layer.output_shape[2], name="%s_up" %name )
                    output = up_dense_layer(output)
                    output = Reshape(dense_layer.input_shape[1:],name="%s_up_reshape"%name)(output)
                    print(up_dense_layer.input_shape,up_dense_layer.output_shape)
                
            output = Add(name="%s_skip"%name)([output,up])

    
        return output


# --------------------------------------------------------------------------------------------------
class DMBuilder(Builder):

    def __init__(self,optimizer=RMSprop,opt_kwargs=dict(lr=0.0002, decay=6e-8)):
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        super(DMBuilder,self).__init__()

    def build(self,discriminator,do_compile=True):
        optimizer = self.optimizer(**self.opt_kwargs)
        if do_compile:
            discriminator.trainable = True
        dm = Model(inputs=discriminator.inputs,outputs=discriminator.outputs)
        if do_compile:
            dm.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
            return dm
        else:
            return dm, optimizer

# --------------------------------------------------------------------------------------------------
class AMBuilder(Builder):

    def __init__(self,optimizer=RMSprop,opt_kwargs=dict(lr=0.0002, decay=6e-8)):
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        super(AMBuilder,self).__init__()

    def build(self,generator,discriminator,do_compile=True):
        optimizer = self.optimizer(**self.opt_kwargs)

        if do_compile:
            discriminator.trainable = False
        wrapped_generator = generator(generator.inputs)
        wrapped_discriminator = discriminator(generator.outputs)
        am = Model(inputs=generator.inputs,outputs=wrapped_discriminator)
        if do_compile:
            am.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
            return am
        else:
            return am,optimizer
        
# --------------------------------------------------------------------------------------------------
class MyFFGAN(MyGAN):

    def __init__(self,x_shape,z_shape,
                 g_opts=dict(),
                 d_opts=dict(),
                 dm_opts=dict(),
                 am_opts=dict(),
                 c_shape=None):

        gBuilder = FFGBuilder(**g_opts)
        dBuilder = FFDBuilder(**d_opts)

        dmBuilder = DMBuilder(**dm_opts)
        amBuilder = AMBuilder(**am_opts)
                
        super(MyFFGAN,self).__init__(x_shape,z_shape,gBuilder,dBuilder,dmBuilder,amBuilder,c_shape)
    
