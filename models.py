import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

from copy import copy

import plotting 

# --------------------------------------------------------------------------------------------------
class Builder(object):

    def __init__(self):

        self._model = None

    def __call__(self,*args,**kwargs):
        if self._model == None:
            self._model = self.build(*args,**kwargs)
        return self._model

    
# --------------------------------------------------------------------------------------------------
class MyGAN(object):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,x_shape,z_shape,gBuilder,dBuilder,dmBuilder,amBuilder,c_shape=None):
        self.x_shape = x_shape
        self.z_shape = z_shape
        self.c_shape = c_shape

        self.gBuilder = gBuilder
        self.dBuilder = dBuilder

        self.amBuilder = amBuilder
        self.dmBuilder = dmBuilder
        
        super(MyGAN,self).__init__()

    # --------------------------------------------------------------------------------------------------
    def get_generator(self):
        return self.gBuilder(self.x_shape,self.z_shape,self.c_shape)
    
    # --------------------------------------------------------------------------------------------------
    def get_discriminator(self):
        return self.dBuilder(self.x_shape,self.c_shape)
    
    # --------------------------------------------------------------------------------------------------
    def compile(self):
        self.dm = self.dmBuilder(self.get_discriminator())
        self.am = self.amBuilder(self.get_generator(),self.get_discriminator())

        return self.am,self.dm
        
    # --------------------------------------------------------------------------------------------------
    def fit(self,
            x_train,z_train,c_x_train=None,c_z_train=None,
            x_test=None,z_test=None,c_x_test=None,c_z_test=None,
            n_disc_steps=1,n_gen_steps=1,
            batch_size=256,n_epochs=50,plot_every=5,print_every=1,solution=None,
    ):
        
        if type( x_test ) == type(None):
            x_test = x_train
        if type( z_test ) == type(None):
            z_test = z_train
        if type( c_x_test ) == type(None):
            c_x_test = c_x_test
        if type( c_z_test ) == type(None):
            c_z_test = c_z_train

        if type( c_z_train ) == type(None):
            c_z_train = c_x_train
        if type( c_z_test ) == type(None):
            c_z_test = c_x_test
            
        has_c = type(c_x_train) != type(None)
        
        self.compile()
        n_batches = x_train.shape[0] // batch_size
        generator = self.get_generator()
        discriminator = self.get_discriminator()
        am = self.am
        dm = self.dm
        print_every = n_batches // print_every
        
        def train_batch(ib):
            x_batch = x_train[ib*batch_size:(ib+1)*batch_size]
            z_batch = z_train[ib*batch_size:(ib+1)*batch_size]
            if has_c:
                c_z_batch = c_z_train[ib*batch_size:(ib+1)*batch_size]
                c_x_batch = c_x_train[ib*batch_size:(ib+1)*batch_size]
                z_batch   = [ c_z_batch, z_batch ]
                g_batch = generator.predict(z_batch)[1]
            else:
                g_batch = generator.predict(z_batch)
            
            x_train_b = np.vstack([x_batch,g_batch])
            if has_c:
                c_train_b = np.vstack([c_x_batch, c_z_batch ])
                x_train_b = [ c_train_b,  x_train_b ]
                
            y_train_b = np.ones( (2*batch_size,1) )
            y_train_b[:batch_size,:] = 0
            
            generator.trainable=False
            for di in range(n_disc_steps):
                d_loss = dm.train_on_batch(x_train_b,y_train_b)
            #d_loss = [0,0]
            generator.trainable=True
            for di in range(n_gen_steps):
                a_loss = am.train_on_batch(z_batch,np.zeros((batch_size,1)))
            # a_loss = [0,0]
            
            if ib % print_every == 0:
                msg = "%d: D: [%f %f] A: [%f %f]" % (ib, d_loss[0], d_loss[1], a_loss[0], a_loss[1])
                print(msg)

        predictions = []
        for iepoch in range(n_epochs):
            if iepoch % plot_every == 0 or iepoch == n_epochs - 1:
                if has_c:
                    x_predict = generator.predict([c_z_test,z_test])[1]
                else:
                    x_predict = generator.predict(z_test)
                predictions.append(x_predict)
                if has_c:
                    x_discrim = discriminator.predict([c_x_test,x_test])
                    z_discrim = discriminator.predict([c_z_test,x_predict])
                    plotting.plot_summary_cond( x_test, c_x_test, x_predict, c_z_test, z_test , x_discrim, z_discrim) #, solution )
                else:
                    x_discrim = discriminator.predict(x_test)
                    z_discrim = discriminator.predict(x_predict)
                    if x_test.shape[-1] == 1:
                        plotting.plot_summary( x_test.ravel(), x_predict.ravel(), z_test.ravel(), x_discrim, z_discrim, solution )
                    else:
                        plotting.plot_summary_2d( x_test, x_predict, x_discrim, z_discrim )
            for ib in range(n_batches):
                train_batch(ib)


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

    def build(self,discriminator):
        optimizer = self.optimizer(**self.opt_kwargs)
        ## dm = Sequential()
        discriminator.trainable = True
        ## dm.add(discriminator)
        dm = Model(inputs=discriminator.inputs,outputs=discriminator.outputs)
        dm.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        
        return dm

# --------------------------------------------------------------------------------------------------
class AMBuilder(Builder):

    def __init__(self,optimizer=RMSprop,opt_kwargs=dict(lr=0.0002, decay=6e-8)):
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        super(AMBuilder,self).__init__()

    def build(self,generator,discriminator):
        optimizer = self.optimizer(**self.opt_kwargs)
        
        ## am = Sequential()
        discriminator.trainable = False
        ## am.add(generator)
        ## am.add(discriminator)
        wrapped_generator = generator(generator.inputs)
        wrapped_discriminator = discriminator(generator.outputs)
        am = Model(inputs=generator.inputs,outputs=wrapped_discriminator)
        am.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

        return am

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
    
