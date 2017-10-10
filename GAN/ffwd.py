import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
import keras.optimizers
from keras.optimizers import Adam, RMSprop

from keras import backend as K

from copy import copy

from .base import Builder, MyGAN, DMBuilder, AMBuilder

from .wgan import WeightClip, wgan_loss


# --------------------------------------------------------------------------------------------------
class FFDBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,kernel_sizes,name="D",activation="sigmoid",clip_weights=None,do_bn=False,do_dropout=False):
        self.kernel_sizes = kernel_sizes
        self.name = name
        self.activation = activation
        self.clip_weights = clip_weights
        self.do_bn = do_bn
        self.do_dropout = do_dropout
        super(FFDBuilder,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,c_shape=None):
        
        do_bn = copy(self.do_bn)
        do_dropout = copy(self.do_dropout)
        input_shape = x_shape
        
        inputs = Input(input_shape,name="%s_input" % self.name)
        if c_shape != None:
            c_inputs = Input(c_shape,name="%s_c_input" % self.name)
            cur = concatenate( [c_inputs,inputs], name = "%s_all_inputs" % self.name )
            inputs = [c_inputs,inputs]
        else:
            cur = inputs
            inputs = [inputs]
        # two paths: w and w/o dropout
        cur = (cur,cur)
        
        ilayer = 1
        for ksize in self.kernel_sizes:
            if do_bn != None:
                bn = do_bn
                if type(do_bn) == list:
                    bn = do_bn.pop(0)
            cur = self.get_unit("%s_down%d" % (self.name,ilayer),cur,ksize,dropout=do_dropout,bn=bn)
            ilayer += 1
            
        flatten_layer = Flatten(name="%s_flat" % self.name)
        flat = (flatten_layer(cur[0]),flatten_layer(cur[1]))
        constraint=None
        if self.clip_weights:
            constraint = WeightClip(self.clip_weights)
        output_layer = Dense(1,activation=self.activation,name="%s_output" % self.name,kernel_constraint=constraint, bias_constraint=constraint)
        output = (output_layer(flat[0]),output_layer(flat[1]))
            
        model = (Model(inputs=inputs,outputs=[output[0]]),Model(inputs=inputs,outputs=[output[1]]))
        return model

    # --------------------------------------------------------------------------------------------------
    def get_unit(self,name,prev,n_out,dropout=False,bn=False):

        constraint=None
        if self.clip_weights:
            constraint = WeightClip(self.clip_weights)
            
        dense_layer = Dense(n_out,use_bias=True,name="%s_dense" % name, kernel_constraint=constraint, bias_constraint=constraint)
        dense = (dense_layer(prev[0]),dense_layer(prev[1]))
        
        if bn:
            batch_norm = BatchNormalization(name="%s_bn" % name,momentum=.5)
            dense = (batch_norm(dense[0]),batch_norm(dense[1]))

        if dropout:
            dense = (Dropout(dropout,name="%s_dropout"%name)(dense[0]),dense[1])
            
        output_layer = Activation("relu",name="%s_activ"%name)
        ## output_layer = Activation("tanh",name="%s_activ"%name)
        ## output_layer = LeakyReLU(name="%s_activ"%name) 
        ## output_layer = PReLU(name="%s_activ"%name)
        output = (output_layer(dense[0]),output_layer(dense[1]))
        
        
        return output


# --------------------------------------------------------------------------------------------------
class FFGBuilder(Builder):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,kernel_sizes,do_down=False,do_skip=False,do_poly=False,do_bn=False,
                 do_nl_activ=False,do_dropout=False,name="G"):
        self.kernel_sizes = kernel_sizes
        self.do_down = do_down
        self.do_skip = do_skip
        self.do_poly = do_poly
        self.do_bn = do_bn
        self.do_nl_activ = do_nl_activ
        self.do_dropout = do_dropout
        self.name = name
        super(FFGBuilder,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def build(self,x_shape,z_shape,c_shape=None):
        
        do_down = copy(self.do_down)
        do_skip = copy(self.do_skip)
        do_poly = copy(self.do_poly)
        do_bn = copy(self.do_bn)
        do_nl_activ = copy(self.do_nl_activ)
        do_dropout = copy(self.do_dropout)
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
                cur = self.get_unit("%s_down%d" % (ilayer,self.name),cur,ksize,dropout=do_dropout,skip=do_skip,bn=do_bn)
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
            cur = self.get_unit("%s_up%d" % (self.name,ilayer),cur,ksize,dropout=do_dropout,skip=do_skip,bn=bn,nl_activ=nl_activ)
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
    def get_unit(self,name,prev,n_out,dropout=False,activate=False,skip=False,bn=False,nl_activ=False):

        inp = prev        

        dense_layer = Dense(n_out,use_bias=True,name="%s_dense" % name)
        dense = dense_layer(prev)

        if bn:
            dense = BatchNormalization(name="%s_bn" % name,momentum=.5)(dense)
    
        if dropout:
            dense = Dropout(dropout,name="%s_dropout"%name)(dense)

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
class MyFFGAN(MyGAN):

    def __init__(self,x_shape,z_shape,
                 g_opts=dict(),
                 d_opts=dict(),
                 dm_opts=dict(),
                 am_opts=dict(),
                 **kwargs
    ):

        gBuilder = FFGBuilder(**g_opts)
        dBuilder = FFDBuilder(**d_opts)

        dmBuilder = DMBuilder(**dm_opts)
        amBuilder = AMBuilder(**am_opts)
                
        super(MyFFGAN,self).__init__(x_shape,z_shape,gBuilder,dBuilder,dmBuilder,amBuilder,**kwargs)
    
