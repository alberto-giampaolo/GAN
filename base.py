import numpy as np

try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3
import plotting 
reload(plotting)

from keras.models import Model

from keras_adversarial import AdversarialOptimizerSimultaneous
from keras_adversarial import AdversarialModel
from keras_adversarial.adversarial_utils import gan_targets

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
    def adversarial_compile(self,loss='binary_crossentropy'):
        dm,dmop = self.dmBuilder(self.get_discriminator(),do_compile=False)
        am,amop = self.amBuilder(self.get_generator(),self.get_discriminator(),do_compile=False)

        self.am = am
        self.dm = dm
        self.gan = Model( inputs=am.inputs + dm.inputs, outputs=am.outputs+dm.outputs )
        
        self.model = AdversarialModel(base_model=self.gan,
                                      player_params=[self.get_generator().trainable_weights,
                                                     self.get_discriminator().trainable_weights],
                                      player_names=["generator", "discriminator"])
        
        self.model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                                       player_optimizers=[amop, dmop],
                                       loss=loss)
        
        
    # --------------------------------------------------------------------------------------------------
    def adversarial_fit(self,
            x_train,z_train,c_x_train=None,c_z_train=None,
            x_test=None,z_test=None,c_x_test=None,c_z_test=None,
            batch_size=256,n_epochs=50,plot_every=5
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

        if has_c:
            train_x = [ c_z_train, z_train, c_x_train, x_train  ]
            test_x = [ c_z_test, z_test, c_x_test, x_test  ]
        else:
            train_x = [ z_train, x_train  ]
            test_x = [ z_test, x_test  ]

        ## train_y = [ np.zeros( (z_train.shape[0],1) ), np.ones( (x_train.shape[0],1) ) ]
        ## test_y = [ np.zeros( (z_test.shape[0],1) ), np.ones( (x_test.shape[0],1) ) ]
        ## validation_data=(test_x,test_y),
        train_y = gan_targets( train_x[0].shape[0] )
        test_y = gan_targets( test_x[0].shape[0] )

        plotter = plotting.SlicePlotter(self.get_generator(),
                               self.get_discriminator(),
                               x_test,z_test,c_x_test,c_z_test,plot_every=plot_every)
        self.model.fit( train_x, train_y,  nb_epoch=n_epochs, batch_size=batch_size,
                        callbacks = [plotter]
        )

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



